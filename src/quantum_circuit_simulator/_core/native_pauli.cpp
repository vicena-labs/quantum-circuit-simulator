// Bounded sparse Pauli-transfer propagation for product-input expectations.
// C ABI permits ctypes calls without a binding-library dependency.
//
// Exact mode drops only roundoff: TRANSFER_ZERO is absolute (PTM entries are
// O(1)); COEFFICIENT_ZERO is scaled by each request's observable l1 norm so
// results scale with the observable. Truncated mode (run_pauli_batch with a
// positive threshold or a weight limit) also discards small or heavy terms and
// returns a rigorous bound on what they could have contributed: a term c*P
// dropped at time t contributes c*Tr(rho_t P), and |Tr(rho_t P)| <= 1.
//
// Terms live in one flat open-addressing table that is updated in place: a
// stage leaves terms with identity on its wires untouched, rescales terms it
// maps to themselves, and rehashes only the terms it moves or branches.
//
// The graph compiler (pauli_graph_*) runs the same propagation once on
// structural support instead of numbers, and records each coefficient update
// as an edge: node[dst] += T[entry] * node[src]. Each node is written by one
// stage and read only by later stages, so an evaluation is one pass over the
// edges, and reverse differentiation is one pass back. A truncated graph also
// propagates coefficients at the bound angles while compiling, chooses the
// terms to discard there, and keeps each discarded term as a node that no
// edge reads: every evaluation then bounds the error from the discarded
// coefficients at its own angles, so the bound holds at any binding.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// The ctypes wrapper loads these entry points by name; Windows exports only
// what is marked, and other platforms may hide symbols by default.
#ifdef _WIN32
#define PAULI_API extern "C" __declspec(dllexport)
#else
#define PAULI_API extern "C" __attribute__((visibility("default")))
#endif

constexpr int K = 17;  // Scalar and at most 16 shared derivatives (sparse solver).
// The graph's reverse pass costs the same for any parameter count; its
// truncated forward tangents grow with it, so the count is still bounded.
constexpr int GRAPH_MAX_PARAMS = 4096;
constexpr double TRANSFER_ZERO = 5e-15;
constexpr double COEFFICIENT_ZERO = 1e-16;

extern "C" {
struct CStage {
    int32_t q0, q1, arity, param;
    double ptm[256];
    double dptm[256];
};
struct CTerm {
    uint64_t x[4], z[4];
    double coef;
};
struct CStats { uint64_t peak_terms, final_terms; };
// One independent backward cone: a selection of stages in forward order, the
// group's own wire numbering for each, and the group's observable terms.
struct CGroup {
    int32_t nqubits, nstages, nterms, reserved;
    const int32_t* indices;      // positions in the full stage array; null = 0..nstages-1
    const int32_t* local_wires;  // two per selected stage (q0, q1); null = stage wires
    const CTerm* terms;          // observable bits in the group's numbering
};
struct CBatchStats {
    uint64_t peak_terms, final_terms, truncated_terms;
    int32_t failed_group, reserved;
    double value_bound;          // |exact - truncated| value, rigorous
    double gradient_bound[16];   // per shared parameter, rigorous
};
// Structural support of one stage for the graph compiler: bit c of rows[r]
// is set when transfer entry (r, c) can be nonzero at some parameter value;
// bit r of identity is set when row r maps to itself with constant value 1.
struct CGraphSupport {
    uint16_t rows[16];
    uint16_t identity, reserved[3];
};
struct CGraphStats {
    uint64_t nodes, edges, peak_terms, bytes, truncated_terms;
    int32_t failed_group, reserved;
};
}

namespace {

inline int popcount64(uint64_t v) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(v));
#else
    return __builtin_popcountll(v);
#endif
}

struct Request {
    int nqubits, nparams, nstages;
    const CStage* full;
    int nfull;
    const int32_t* indices;
    const int32_t* local_wires;
    int nterms;
    const CTerm* observables;
    int initial_mode;
    uint64_t max_terms;
    double truncate_below;      // absolute; 0 disables
    int max_weight;             // negative disables
    const double* lipschitz;    // per full stage; bounds d<P>/dtheta for truncated gradients

    int stage_index(int step) const { return indices ? indices[step] : step; }
    CStage const& stage(int step) const { return full[stage_index(step)]; }
    void wires(int step, int& q0, int& q1) const {
        CStage const& s = stage(step);
        q0 = local_wires ? local_wires[2 * step] : s.q0;
        q1 = s.arity == 2 ? (local_wires ? local_wires[2 * step + 1] : s.q1) : -1;
    }
};

// Each wire's earliest stage in forward order, and the declared and
// never-touched wire masks. Immediately after crossing a wire's earliest stage
// backwards, the wire can be contracted into the known product input: no
// earlier remaining operation can touch it. This applies to derivative
// coefficients as well as the value, including affine T1.
template<int W>
struct ConeLayout {
    std::array<int, 256> first_touch;
    std::array<uint64_t, W> declared{}, untouched{};

    // False on an invalid stage selection or wire numbering.
    bool build(Request const& rq) {
        first_touch.fill(-1);
        for (int step = 0; step < rq.nstages; ++step) {
            if (rq.indices && (rq.indices[step] < 0 || rq.indices[step] >= rq.nfull)) return false;
            CStage const& stage = rq.stage(step);
            int q0, q1;
            rq.wires(step, q0, q1);
            if (stage.arity < 1 || stage.arity > 2 || q0 < 0 || q0 >= rq.nqubits ||
                (stage.arity == 2 && (q1 < 0 || q1 >= rq.nqubits || q1 == q0)) ||
                stage.param < -1 || stage.param >= rq.nparams) return false;
            if (first_touch[q0] < 0) first_touch[q0] = step;
            if (stage.arity == 2 && first_touch[q1] < 0) first_touch[q1] = step;
        }
        for (int q = 0; q < rq.nqubits; ++q) {
            const uint64_t bit = 1ULL << (q & 63);
            declared[q >> 6] |= bit;
            if (first_touch[q] < 0) untouched[q >> 6] |= bit;
        }
        return true;
    }

    // Key of an observable term with never-touched wires contracted: 1 keeps
    // it, 0 drops it (zero overlap with the input), -2 rejects invalid bits.
    int initial_key(CTerm const& term, int initial_mode, uint64_t* key) const {
        if (!std::isfinite(term.coef)) return -2;
        int keep = term.coef != 0.;
        for (int j = 0; j < 4; ++j) {
            const uint64_t x = term.x[j], z = term.z[j];
            if (j >= W) {
                if (x | z) return -2;
                continue;
            }
            if ((x | z) & ~declared[j]) return -2;
            if ((initial_mode == 0 ? x : z) & untouched[j]) keep = 0;
            key[j] = x & ~untouched[j];
            key[W + j] = z & ~untouched[j];
        }
        return keep;
    }
};

struct Result {
    int code = 0;
    uint64_t peak_terms = 0, final_terms = 0, truncated_terms = 0;
    double value_bound = 0.;
    std::array<double, 16> gradient_bound{};
    std::array<double, K> out{};
};

// Flat table of Pauli keys (x words then z words) with a fixed stride of
// values per key: coefficients in the solver, node numbers in the graph
// compiler. Slot states: empty, live, deleted, live and changed by this stage.
template<int W, typename V = double>
class TermTable {
public:
    static constexpr int KW = 2 * W;
    enum : uint8_t { EMPTY = 0, LIVE = 1, DELETED = 2, CHANGED = 3 };

    explicit TermTable(int stride) : stride(stride) {}

    size_t capacity() const { return state.size(); }

    // Slot of `key`, inserting zeroed coefficients if absent. Returns SIZE_MAX
    // when inserting would exceed `limit` live terms. Live and deleted slots
    // stay below 70% of capacity; a rehash leaves the table at most half full.
    size_t find_or_insert(const uint64_t* key, uint64_t limit) {
        if ((size + deleted + 1) * 10 > capacity() * 7) {
            size_t cap = 16;
            while (cap < (size + 1) * 2) cap <<= 1;
            rehash(cap);
        }
        const size_t mask = capacity() - 1;
        size_t i = hash(key) & mask, tomb = SIZE_MAX;
        while (true) {
            const uint8_t s = state[i];
            if (s == EMPTY) break;
            if (s == DELETED) {
                if (tomb == SIZE_MAX) tomb = i;
            } else if (std::memcmp(&keys[i * KW], key, sizeof(uint64_t) * KW) == 0) {
                return i;
            }
            i = (i + 1) & mask;
        }
        if (size >= limit) return SIZE_MAX;
        if (tomb != SIZE_MAX) { i = tomb; --deleted; }
        std::memcpy(&keys[i * KW], key, sizeof(uint64_t) * KW);
        std::fill_n(&coefs[i * stride], stride, V{});
        state[i] = CHANGED;
        ++size;
        return i;
    }

    void erase(size_t slot) { state[slot] = DELETED; --size; ++deleted; }

    std::vector<uint8_t> state;
    std::vector<uint64_t> keys;
    std::vector<V> coefs;
    size_t size = 0, deleted = 0;
    int stride;

private:
    static uint64_t hash(const uint64_t* key) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int j = 0; j < KW; ++j) {
            h ^= key[j];
            h *= 0xbf58476d1ce4e5b9ULL;
            h ^= h >> 31;
        }
        h *= 0x94d049bb133111ebULL;
        return h ^ (h >> 29);
    }

    void rehash(size_t cap) {
        std::vector<uint8_t> old_state(cap, EMPTY);
        std::vector<uint64_t> old_keys(cap * KW);
        std::vector<V> old_coefs(cap * stride);
        old_state.swap(state);
        old_keys.swap(keys);
        old_coefs.swap(coefs);
        const size_t mask = cap - 1;
        for (size_t s = 0; s < old_state.size(); ++s) {
            if (old_state[s] != LIVE && old_state[s] != CHANGED) continue;
            size_t i = hash(&old_keys[s * KW]) & mask;
            while (state[i] != EMPTY) i = (i + 1) & mask;
            state[i] = old_state[s];
            std::memcpy(&keys[i * KW], &old_keys[s * KW], sizeof(uint64_t) * KW);
            std::memcpy(&coefs[i * stride], &old_coefs[s * stride], sizeof(V) * stride);
        }
        deleted = 0;
    }
};

inline bool nonzero_input_overlap(int label, int initial_mode) {
    // <0|I,Z|0> = 1; <+|I,X|+> = 1. All other labels vanish.
    return (label & (initial_mode == 0 ? 1 : 2)) == 0;
}

struct Transition {
    int label;
    double value, derivative;
};

// One row of a stage's transfer matrix, specialised once per stage.
struct Row {
    int size = -1;      // -1: not built yet
    bool identity = false;  // maps the term to itself with value 1, no tangent
    bool diagonal = false;  // maps the term to itself only
    std::array<Transition, 16> transitions;
};

inline int lowest_bit(uint64_t v) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, v);
    return static_cast<int>(index);
#else
    return __builtin_ctzll(v);
#endif
}

// Specialise one row of a dim x dim transfer matrix: the labels it reaches,
// with wires whose earliest stage this is projected onto the product input.
bool build_row(Row& r, int row, const double* ptm, const double* dptm, int dim, bool tangent,
               bool close0, bool close1, int initial_mode) {
    int count = 0;
    for (int label = 0; label < dim; ++label) {
        const double value = ptm[row * dim + label];
        const double derivative = tangent ? dptm[row * dim + label] : 0.;
        if (!std::isfinite(value) || !std::isfinite(derivative)) return false;
        // dt-only transitions at an exact zero of t still carry real
        // first derivatives and must remain in this union.
        if (std::abs(value) <= TRANSFER_ZERO && std::abs(derivative) <= TRANSFER_ZERO) continue;
        if ((close0 && !nonzero_input_overlap(label & 3, initial_mode)) ||
            (close1 && !nonzero_input_overlap(label >> 2, initial_mode))) continue;
        const int projected = (close0 ? 0 : (label & 3)) | (close1 ? 0 : (label & 12));
        // Closing a wire maps several labels to one; sum them here, or a
        // term crossing many closing stages of a layer would expand into
        // exponentially many copies of the same image.
        int j = 0;
        while (j < count && r.transitions[j].label != projected) ++j;
        if (j < count) {
            r.transitions[j].value += value;
            r.transitions[j].derivative += derivative;
        } else {
            r.transitions[count++] = {projected, value, derivative};
        }
    }
    r.size = count;
    r.diagonal = count == 1 && r.transitions[0].label == row;
    r.identity = r.diagonal && r.transitions[0].value == 1. && r.transitions[0].derivative == 0.;
    return true;
}

// The exact solver's unit of work: one gate merged with the channels and
// single-wire gates beside it on the same wires, in the cone's own wire
// numbering. At most one merged stage carries a shared parameter, so the
// merged derivative is one product. `last_step` is the latest original stage
// it contains, which places truncation cuts for the gradient bound.
struct Fused {
    int q0, q1, arity, param, last_step;
    std::array<double, 256> ptm, dptm;
};

// `m` (a stage on wires a, b; b < 0 for one wire) as a 16x16 transfer matrix
// on the merged stage's wires f0, f1, whose labels are l0 | l1 << 2.
void embed(const double* m, int arity, int a, int b, int f0, double* out) {
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            const int r0 = r & 3, r1 = r >> 2, c0 = c & 3, c1 = c >> 2;
            double v;
            if (arity == 1) v = a == f0 ? (r1 == c1 ? m[r0 * 4 + c0] : 0.) : (r0 == c0 ? m[r1 * 4 + c1] : 0.);
            else if (a == f0) v = m[r * 16 + c];
            else v = m[(r1 | r0 << 2) * 16 + (c1 | c0 << 2)];
            out[r * 16 + c] = v;
        }
    (void)b;
}

// Transfer matrices of gates and channels are mostly zeros, so the product
// skips them: merging costs a few operations per stage, not dim^3.
void multiply(const double* x, const double* y, int dim, double* out) {
    std::fill_n(out, dim * dim, 0.);
    for (int r = 0; r < dim; ++r)
        for (int k = 0; k < dim; ++k) {
            const double v = x[r * dim + k];
            if (v == 0.) continue;
            for (int c = 0; c < dim; ++c) out[r * dim + c] += v * y[k * dim + c];
        }
}

// f becomes `later` applied after f (later = false: `earlier` applied
// before f). Heisenberg propagation meets the later operation first, so the
// merged transfer matrix is T_later * T_earlier.
void compose(Fused& f, const double* ptm, const double* dptm, int param, int arity, int a, int b,
             bool later) {
    const int dim = f.arity == 1 ? 4 : 16;
    std::array<double, 256> m, dm, product;
    if (f.arity == 1) {
        std::copy_n(ptm, 16, m.begin());
        std::copy_n(dptm, 16, dm.begin());
    } else {
        embed(ptm, arity, a, b, f.q0, m.data());
        embed(dptm, arity, a, b, f.q0, dm.data());
    }
    if (later) {
        if (param >= 0) multiply(dm.data(), f.ptm.data(), dim, f.dptm.data());
        else if (f.param >= 0) { multiply(m.data(), f.dptm.data(), dim, product.data()); f.dptm = product; }
        multiply(m.data(), f.ptm.data(), dim, product.data());
    } else {
        if (param >= 0) multiply(f.ptm.data(), dm.data(), dim, f.dptm.data());
        else if (f.param >= 0) { multiply(f.dptm.data(), m.data(), dim, product.data()); f.dptm = product; }
        multiply(f.ptm.data(), m.data(), dim, product.data());
    }
    f.ptm = product;
    if (param >= 0) f.param = param;
}

// Merge the cone's stages: a single-wire stage joins the latest merged stage
// on its wire, and a two-wire stage joins the latest one on the same pair or
// absorbs the single-wire stages that end on its wires. Moving a stage this
// way is exact, because no stage between them touches its wires.
// Absorbed stages stay in place with `alive` cleared, which avoids copying
// every merged matrix again.
void fuse(Request const& rq, std::vector<Fused>& fused, std::vector<char>& alive) {
    fused.clear();
    alive.clear();
    fused.reserve(rq.nstages);
    std::array<int, 256> last;
    last.fill(-1);
    // Only the dim x dim block of each matrix is used, so only it is filled.
    auto fresh = [&](CStage const& s, int q0, int q1, int step) {
        Fused f;
        f.q0 = q0, f.q1 = q1, f.arity = s.arity, f.param = s.param, f.last_step = step;
        const int n = s.arity == 1 ? 16 : 256;
        std::copy_n(s.ptm, n, f.ptm.begin());
        if (s.param >= 0) std::copy_n(s.dptm, n, f.dptm.begin());
        else std::fill_n(f.dptm.begin(), n, 0.);
        return f;
    };
    for (int step = 0; step < rq.nstages; ++step) {
        CStage const& s = rq.stage(step);
        int q0, q1;
        rq.wires(step, q0, q1);
        if (s.arity == 1) {
            const int p = last[q0];
            if (p >= 0 && !(fused[p].param >= 0 && s.param >= 0)) {
                compose(fused[p], s.ptm, s.dptm, s.param, 1, q0, -1, true);
                fused[p].last_step = step;
                continue;
            }
            last[q0] = static_cast<int>(fused.size());
            fused.push_back(fresh(s, q0, q1, step));
            alive.push_back(1);
            continue;
        }
        const int pa = last[q0], pb = last[q1];
        if (pa >= 0 && pa == pb && !(fused[pa].param >= 0 && s.param >= 0)) {
            compose(fused[pa], s.ptm, s.dptm, s.param, 2, q0, q1, true);
            fused[pa].last_step = step;
            continue;
        }
        Fused f = fresh(s, q0, q1, step);
        for (int p : {pa, pb}) {
            if (p < 0 || !alive[p] || fused[p].arity != 1 || (fused[p].param >= 0 && f.param >= 0)) continue;
            compose(f, fused[p].ptm.data(), fused[p].dptm.data(), fused[p].param, 1, fused[p].q0, -1, false);
            alive[p] = 0;
        }
        last[q0] = last[q1] = static_cast<int>(fused.size());
        fused.push_back(f);
        alive.push_back(1);
    }
}

// Propagation by layers. Merged stages are scheduled backwards as soon as
// every later stage on their wires has run, so each layer's stages act on
// distinct wires and commute. One pass over the table per layer finds the
// terms a layer changes; each such term goes through all of its stages at
// once and is inserted once per image, instead of once per stage. Terms are
// filtered after each layer at a cut just past the latest original stage not
// yet applied, so a truncated gradient's bound covers every stage still ahead
// of a discarded term (and a few already behind it, which only loosens it).
template<int W, bool WithGradient, typename Filter, typename Discard>
bool propagate_layers(Request const& rq, TermTable<W>& terms, Filter& filter, Discard& discard,
                      Result& result) {
    using Table = TermTable<W>;
    constexpr int KW = Table::KW;
    const int noutputs = WithGradient ? rq.nparams + 1 : 1;
    std::vector<Fused> stages;
    std::vector<char> alive;
    fuse(rq, stages, alive);
    const int n = static_cast<int>(stages.size());

    std::array<int, 256> first_touch, level;
    first_touch.fill(-1);
    level.fill(-1);
    for (int i = 0; i < n; ++i) {
        if (!alive[i]) continue;
        Fused const& f = stages[i];
        if (first_touch[f.q0] < 0) first_touch[f.q0] = i;
        if (f.arity == 2 && first_touch[f.q1] < 0) first_touch[f.q1] = i;
    }
    std::vector<std::vector<int>> layers;
    for (int i = n - 1; i >= 0; --i) {
        if (!alive[i]) continue;
        Fused const& f = stages[i];
        const int l = 1 + std::max(level[f.q0], f.arity == 2 ? level[f.q1] : -1);
        level[f.q0] = l;
        if (f.arity == 2) level[f.q1] = l;
        if (l >= static_cast<int>(layers.size())) layers.resize(l + 1);
        layers[l].push_back(i);
    }

    // cut_from[l] lies just past every original stage in layers l onwards;
    // terms discarded after layer l use cut_from[l + 1], and branches
    // discarded inside layer l use cut_from[l].
    std::vector<int> cut_from(layers.size() + 1, 0);
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
        int latest = cut_from[l + 1] - 1;
        for (int i : layers[l]) latest = std::max(latest, stages[i].last_step);
        cut_from[l] = latest + 1;
    }
    // In truncated mode a branch below the threshold is dropped as soon as a
    // stage produces it, so a layer never holds more than per-stage
    // truncation would.
    const double prune_below = rq.truncate_below;

    struct Touch { int stage; int row; };
    std::array<int, 256> owner;
    owner.fill(-1);
    std::vector<std::array<Row, 16>> rows;
    std::vector<uint32_t> seen;
    std::vector<Touch> touches, pending_touches;
    std::vector<uint64_t> pending_keys, stack_keys;
    std::vector<double> pending_coefs, stack_coefs;
    std::vector<uint32_t> pending_start;

    for (size_t l = 0; l < layers.size(); ++l) {
        std::vector<int> const& layer = layers[l];
        const int m = static_cast<int>(layer.size());
        std::array<uint64_t, W> mask{};
        // Reset only the row sizes: value-initialising every transition
        // dominated small circuits.
        if (static_cast<int>(rows.size()) < m) rows.resize(m);
        for (int i = 0; i < m; ++i)
            for (Row& r : rows[i]) r.size = -1;
        seen.assign(m, 0);
        for (int i = 0; i < m; ++i) {
            Fused const& f = stages[layer[i]];
            owner[f.q0] = i;
            mask[f.q0 >> 6] |= 1ULL << (f.q0 & 63);
            if (f.arity == 2) {
                owner[f.q1] = i;
                mask[f.q1 >> 6] |= 1ULL << (f.q1 & 63);
            }
        }
        auto label = [](const uint64_t* k, int q) {
            const uint64_t b = 1ULL << (q & 63);
            return ((k[q >> 6] & b) ? 1 : 0) + ((k[W + (q >> 6)] & b) ? 2 : 0);
        };
        // The stage's row for this key, built on first use.
        auto row_of = [&](int i, const uint64_t* k, Row*& r) -> int {
            Fused const& f = stages[layer[i]];
            int row = label(k, f.q0);
            if (f.arity == 2) row += label(k, f.q1) << 2;
            r = &rows[i][row];
            if (r->size < 0 &&
                !build_row(*r, row, f.ptm.data(), f.dptm.data(), f.arity == 1 ? 4 : 16,
                           WithGradient && f.param >= 0, first_touch[f.q0] == layer[i],
                           f.arity == 2 && first_touch[f.q1] == layer[i], rq.initial_mode))
                return -1;
            return row;
        };

        // Pass 1: rescale terms that only meet diagonal rows in place, and
        // move out the terms some stage of the layer changes.
        pending_keys.clear();
        pending_coefs.clear();
        pending_touches.clear();
        pending_start.clear();
        uint32_t stamp = 0;
        for (size_t s = 0; s < terms.capacity(); ++s) {
            const uint8_t st = terms.state[s];
            if (st != Table::LIVE && st != Table::CHANGED) continue;
            const uint64_t* k = &terms.keys[s * KW];
            touches.clear();
            bool moves = false;
            ++stamp;
            for (int j = 0; j < W; ++j) {
                uint64_t bits = (k[j] | k[W + j]) & mask[j];
                while (bits) {
                    const int q = j * 64 + lowest_bit(bits);
                    bits &= bits - 1;
                    const int i = owner[q];
                    if (seen[i] == stamp) continue;
                    seen[i] = stamp;
                    Row* r;
                    const int row = row_of(i, k, r);
                    if (row < 0) { result.code = -2; return false; }
                    if (r->identity) continue;
                    touches.push_back({i, row});
                    if (!r->diagonal) moves = true;
                }
            }
            if (touches.empty()) continue;
            double* c = &terms.coefs[s * terms.stride];
            if (!moves) {
                for (Touch const& t : touches) {
                    Transition const& tr = rows[t.stage][t.row].transitions[0];
                    const double c0 = c[0];
                    for (int j = 0; j < noutputs; ++j) c[j] *= tr.value;
                    const int param = stages[layer[t.stage]].param;
                    if (WithGradient && param >= 0) c[param + 1] += tr.derivative * c0;
                }
                terms.state[s] = Table::CHANGED;
                continue;
            }
            bool vanishes = false;
            for (Touch const& t : touches) vanishes |= rows[t.stage][t.row].size == 0;
            if (!vanishes) {
                pending_start.push_back(static_cast<uint32_t>(pending_touches.size()));
                pending_touches.insert(pending_touches.end(), touches.begin(), touches.end());
                pending_keys.insert(pending_keys.end(), k, k + KW);
                pending_coefs.insert(pending_coefs.end(), c, c + noutputs);
            }
            terms.erase(s);
        }
        pending_start.push_back(static_cast<uint32_t>(pending_touches.size()));

        // Pass 2: expand each moved term through its stages, depth first,
        // and merge its images. Exceeding the hard cap always fails.
        stack_keys.resize(static_cast<size_t>(m + 1) * KW);
        stack_coefs.resize(static_cast<size_t>(m + 1) * noutputs);
        const size_t moved = pending_start.size() - 1;
        for (size_t p = 0; p < moved; ++p) {
            const Touch* touch = &pending_touches[pending_start[p]];
            const int depth_end = static_cast<int>(pending_start[p + 1] - pending_start[p]);
            std::copy_n(&pending_keys[p * KW], KW, stack_keys.begin());
            std::copy_n(&pending_coefs[p * noutputs], noutputs, stack_coefs.begin());
            auto expand = [&](auto&& self, int depth) -> bool {
                uint64_t* k = &stack_keys[static_cast<size_t>(depth) * KW];
                double* c = &stack_coefs[static_cast<size_t>(depth) * noutputs];
                if (depth == depth_end) {
                    const size_t slot = terms.find_or_insert(k, rq.max_terms);
                    if (slot == SIZE_MAX) return false;
                    double* destination = &terms.coefs[slot * terms.stride];
                    for (int j = 0; j < noutputs; ++j) destination[j] += c[j];
                    terms.state[slot] = Table::CHANGED;
                    return true;
                }
                Fused const& f = stages[layer[touch[depth].stage]];
                Row const& r = rows[touch[depth].stage][touch[depth].row];
                const bool tangent = WithGradient && f.param >= 0;
                uint64_t* next_key = k + KW;
                double* next = c + noutputs;
                const int w0 = f.q0 >> 6, w1 = f.q1 >> 6;
                const uint64_t b0 = 1ULL << (f.q0 & 63), b1 = f.arity == 2 ? 1ULL << (f.q1 & 63) : 0;
                for (int i = 0; i < r.size; ++i) {
                    Transition const& t = r.transitions[i];
                    std::copy_n(k, KW, next_key);
                    next_key[w0] = (next_key[w0] & ~b0) | ((t.label & 1) ? b0 : 0);
                    next_key[W + w0] = (next_key[W + w0] & ~b0) | ((t.label & 2) ? b0 : 0);
                    if (f.arity == 2) {
                        next_key[w1] = (next_key[w1] & ~b1) | ((t.label & 4) ? b1 : 0);
                        next_key[W + w1] = (next_key[W + w1] & ~b1) | ((t.label & 8) ? b1 : 0);
                    }
                    double magnitude = 0.;
                    for (int j = 0; j < noutputs; ++j) {
                        next[j] = t.value * c[j];
                        magnitude = std::max(magnitude, std::abs(next[j]));
                    }
                    if (tangent) {
                        next[f.param + 1] += t.derivative * c[0];
                        magnitude = std::max(magnitude, std::abs(next[f.param + 1]));
                    }
                    if (magnitude < prune_below) {
                        discard(next, cut_from[l]);
                        continue;
                    }
                    if (!self(self, depth + 1)) return false;
                }
                return true;
            };
            if (!expand(expand, 0)) { result.code = -1; return false; }
        }
        for (int i = 0; i < m; ++i) {
            Fused const& f = stages[layer[i]];
            owner[f.q0] = -1;
            if (f.arity == 2) owner[f.q1] = -1;
        }
        // The table is largest now, before this layer's discards.
        result.peak_terms = std::max<uint64_t>(result.peak_terms, terms.size);
        if (!filter(cut_from[l + 1])) { result.code = -2; return false; }
    }
    return true;
}

template<int W, bool WithGradient>
void run_cone(Request const& rq, Result& result) {
    using Table = TermTable<W>;
    constexpr int KW = Table::KW;
    const int noutputs = WithGradient ? rq.nparams + 1 : 1;
    const bool truncating = rq.truncate_below > 0. || rq.max_weight >= 0;
    ConeLayout<W> layout;
    if (!layout.build(rq)) { result.code = -2; return; }
    // Bound on |d Tr(rho_t P)/d theta_j| from the parameterised stages before
    // cut t: prefix sums of each stage's 2*||generator||.
    std::vector<double> lipschitz_before;
    if (WithGradient && truncating && rq.nparams > 0) {
        if (!rq.lipschitz) { result.code = -2; return; }
        lipschitz_before.assign(static_cast<size_t>(rq.nstages + 1) * rq.nparams, 0.);
        for (int step = 0; step < rq.nstages; ++step) {
            double* next = &lipschitz_before[static_cast<size_t>(step + 1) * rq.nparams];
            std::copy_n(next - rq.nparams, rq.nparams, next);
            CStage const& stage = rq.stage(step);
            const double l = rq.lipschitz[rq.stage_index(step)];
            if (!std::isfinite(l) || l < 0.) { result.code = -2; return; }
            if (stage.param >= 0) next[stage.param] += l;
        }
    }
    Table terms(noutputs);
    std::array<uint64_t, KW> key;
    for (int i = 0; i < rq.nterms; ++i) {
        CTerm const& term = rq.observables[i];
        const int keep = layout.initial_key(term, rq.initial_mode, key.data());
        if (keep < 0) { result.code = -2; return; }
        if (!keep) continue;
        const size_t slot = terms.find_or_insert(key.data(), rq.max_terms);
        if (slot == SIZE_MAX) { result.code = -1; return; }
        double& c = terms.coefs[slot * terms.stride];
        c += term.coef;
        if (!std::isfinite(c)) { result.code = -2; return; }
    }
    result.peak_terms = terms.size;
    // A coefficient is roundoff only relative to the observable it came
    // from: an absolute cutoff would silently zero small-weight observables.
    double observable_norm = 0.;
    for (size_t s = 0; s < terms.capacity(); ++s)
        if (terms.state[s] != Table::EMPTY && terms.state[s] != Table::DELETED)
            observable_norm += std::abs(terms.coefs[s * terms.stride]);
    const double coefficient_zero = COEFFICIENT_ZERO * observable_norm;

    // Add a discarded coefficient to the error bounds. `cut` is the number of
    // stages that may still act on it. Roundoff drops are counted too, so the
    // bound covers every discarded coefficient.
    auto discard = [&](const double* c, int cut) {
        if (!truncating) return;
        const double* before = lipschitz_before.empty() ? nullptr
            : &lipschitz_before[static_cast<size_t>(cut) * rq.nparams];
        ++result.truncated_terms;
        result.value_bound += std::abs(c[0]);
        if constexpr (WithGradient)
            for (int j = 0; j < rq.nparams; ++j)
                result.gradient_bound[j] += std::abs(c[j + 1]) + (before ? std::abs(c[0]) * before[j] : 0.);
    };
    // Drop roundoff and, when truncating, small or heavy terms among those
    // changed since the last pass.
    auto filter = [&](int cut) -> bool {
        for (size_t s = 0; s < terms.capacity(); ++s) {
            if (terms.state[s] != Table::CHANGED) continue;
            double* c = &terms.coefs[s * terms.stride];
            double magnitude = 0.;
            for (int j = 0; j < noutputs; ++j) {
                if (!std::isfinite(c[j])) return false;
                magnitude = std::max(magnitude, std::abs(c[j]));
            }
            bool drop = magnitude <= coefficient_zero;
            if (!drop && truncating) {
                drop = magnitude < rq.truncate_below;
                if (!drop && rq.max_weight >= 0) {
                    int weight = 0;
                    const uint64_t* k = &terms.keys[s * KW];
                    for (int j = 0; j < W; ++j) weight += popcount64(k[j] | k[W + j]);
                    drop = weight > rq.max_weight;
                }
            }
            if (!drop) { terms.state[s] = Table::LIVE; continue; }
            discard(c, cut);
            terms.erase(s);
        }
        return true;
    };
    if (truncating && !filter(rq.nstages)) { result.code = -2; return; }

    if (!propagate_layers<W, WithGradient>(rq, terms, filter, discard, result)) return;
    // All declared wires have already been contracted, including circuits
    // with no stages. Retain the overlap check as a defensive invariant.
    for (size_t s = 0; s < terms.capacity(); ++s) {
        const uint8_t st = terms.state[s];
        if (st != Table::LIVE && st != Table::CHANGED) continue;
        const uint64_t* k = &terms.keys[s * KW];
        bool keep = true;
        for (int j = 0; j < W; ++j)
            if ((rq.initial_mode == 0 ? k[j] : k[W + j]) != 0) keep = false;
        if (!keep) continue;
        for (int j = 0; j < noutputs; ++j) result.out[j] += terms.coefs[s * terms.stride + j];
    }
    for (int j = 0; j < noutputs; ++j)
        if (!std::isfinite(result.out[j])) { result.code = -2; return; }
    result.final_terms = terms.size;
}

template<bool WithGradient>
Result run_request(Request const& rq) {
    Result result;
    if (rq.nqubits < 1 || rq.nqubits > 256 || rq.nparams < 0 || rq.nparams > 16 ||
        rq.nstages < 0 || rq.nterms < 0 || rq.max_terms < 1 || rq.nfull < 0 ||
        (rq.nterms && !rq.observables) || (rq.nstages && !rq.full) ||
        (!rq.indices && rq.nstages > rq.nfull) ||
        !(rq.truncate_below >= 0.) || !std::isfinite(rq.truncate_below) ||
        rq.max_weight < -1 || (rq.initial_mode != 0 && rq.initial_mode != 1)) {
        result.code = -2;
        return result;
    }
    try {
        const int words = (rq.nqubits + 63) / 64;
        if (words == 1) run_cone<1, WithGradient>(rq, result);
        else if (words == 2) run_cone<2, WithGradient>(rq, result);
        else if (words == 3) run_cone<3, WithGradient>(rq, result);
        else run_cone<4, WithGradient>(rq, result);
    } catch (std::bad_alloc const&) { result.code = -3; }
      catch (...) { result.code = -4; }
    return result;
}

// Original single-cone ABI: no truncation, the caller's wire numbering.
template<bool WithGradient>
int run_single(int nqubits, int nparams, int nstages, const CStage* full_stages, int nfull,
               const int32_t* indices, int nterms, const CTerm* observables, int initial_mode,
               uint64_t max_terms, double* out, CStats* stats) {
    if (!out || !stats || (nstages > 0 && !full_stages)) return -2;
    if (indices && (nfull < 0 || (nfull > 0 && !full_stages))) return -2;
    Request rq{nqubits, nparams, nstages, full_stages, indices ? nfull : nstages, indices,
               nullptr, nterms, observables, initial_mode, max_terms, 0., -1, nullptr};
    Result result = run_request<WithGradient>(rq);
    if (result.code) return result.code;
    const int noutputs = WithGradient ? nparams + 1 : 1;
    std::copy_n(result.out.begin(), noutputs, out);
    stats->peak_terms = result.peak_terms;
    stats->final_terms = result.final_terms;
    return 0;
}


// Run `body(g)` for g in [0, n) on up to `nthreads` threads. Workers stop
// taking new items once one fails, so the lowest failure is always run.
template<typename Body>
int parallel_for(int n, int nthreads, Body body) {
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    auto work = [&]() {
        while (!failed.load(std::memory_order_relaxed)) {
            const int g = next.fetch_add(1);
            if (g >= n) break;
            if (!body(g)) failed = true;
        }
    };
    const int workers = std::min(nthreads, n);
    if (workers <= 1) { work(); return 0; }
    try {
        std::vector<std::thread> pool;
        for (int t = 1; t < workers; ++t) pool.emplace_back(work);
        work();
        for (auto& thread : pool) thread.join();
    } catch (...) { return -4; }
    return 0;
}

// Edges [begin, end) all read one stage's transfer matrix.
struct GraphLayer {
    int32_t stage;
    uint32_t begin, end;
};

struct Graph {
    int code = 0;
    uint32_t nodes = 0;
    uint64_t peak_terms = 0, truncated_terms = 0;
    bool truncating = false;
    std::vector<double> initial;   // values of nodes 0 .. initial.size()-1
    std::vector<GraphLayer> layers;
    std::vector<int32_t> source, destination;
    std::vector<uint8_t> entry;    // row * dim + column in the stage's matrix
    std::vector<int32_t> outputs;  // nodes summed into the result
    // Truncation: each discarded node, the number of the group's stages
    // before it was discarded, and per such cut the sum of each parameter's
    // stage bounds 2*||G|| before it (row-major, nparams per cut).
    std::vector<int32_t> dropped, dropped_cut;
    std::vector<double> lipschitz_before;
    std::vector<double> values, adjoint;  // evaluation scratch, reused

    uint64_t bytes() const {
        return initial.size() * 8 + layers.size() * sizeof(GraphLayer) + source.size() * 9 +
               outputs.size() * 4 + dropped.size() * 8 + lipschitz_before.size() * 8 +
               static_cast<uint64_t>(nodes) * 16;
    }
};

struct GraphSet {
    int nfull;
    std::vector<Graph> groups;
};

struct GraphLimits {
    uint64_t max_terms, max_nodes, max_edges;
};

// Graph codes beyond the solver's: node budget, edge budget, an uncontracted
// final key.
constexpr int GRAPH_NODES = -7, GRAPH_EDGES = -8, GRAPH_BOUNDARY = -6;

struct GraphRow {
    int size = -1;
    bool identity = false;
    std::array<uint8_t, 16> label, entry;
};

template<int W>
void compile_graph(Request const& rq, const CGraphSupport* support, GraphLimits limits, Graph& g) {
    constexpr int KW = 2 * W;
    ConeLayout<W> layout;
    if (!layout.build(rq)) { g.code = -2; return; }
    g.truncating = rq.truncate_below > 0. || rq.max_weight >= 0;
    // Table slots hold node + 1, then, when truncating, the value and its
    // tangents at the bound angles.
    const int ncoefs = rq.nparams + 1;
    const int stride = g.truncating ? 1 + ncoefs : 1;
    if (g.truncating && rq.nparams > 0) {
        if (!rq.lipschitz) { g.code = -2; return; }
        g.lipschitz_before.assign(static_cast<size_t>(rq.nstages + 1) * rq.nparams, 0.);
        for (int step = 0; step < rq.nstages; ++step) {
            double* next = &g.lipschitz_before[static_cast<size_t>(step + 1) * rq.nparams];
            std::copy_n(next - rq.nparams, rq.nparams, next);
            const double l = rq.lipschitz[rq.stage_index(step)];
            if (!std::isfinite(l) || l < 0.) { g.code = -2; return; }
            if (rq.stage(step).param >= 0) next[rq.stage(step).param] += l;
        }
    }
    // Observable coefficients merge per contracted key; exact zeros vanish.
    TermTable<W> sums(1);
    std::array<uint64_t, KW> key;
    for (int i = 0; i < rq.nterms; ++i) {
        const int keep = layout.initial_key(rq.observables[i], rq.initial_mode, key.data());
        if (keep < 0) { g.code = -2; return; }
        if (!keep) continue;
        const size_t slot = sums.find_or_insert(key.data(), limits.max_terms);
        if (slot == SIZE_MAX) { g.code = -1; return; }
        sums.coefs[slot] += rq.observables[i].coef;
    }
    TermTable<W> terms(stride);
    for (size_t s = 0; s < sums.capacity(); ++s) {
        if (sums.state[s] == TermTable<W>::EMPTY || sums.state[s] == TermTable<W>::DELETED) continue;
        const double c = sums.coefs[s];
        if (!std::isfinite(c)) { g.code = -2; return; }
        if (c == 0.) continue;
        double* slot = &terms.coefs[terms.find_or_insert(&sums.keys[s * KW], limits.max_terms) * stride];
        slot[0] = ++g.nodes;
        if (g.truncating) slot[1] = c;
        g.initial.push_back(c);
    }
    if (g.nodes > limits.max_nodes) { g.code = GRAPH_NODES; return; }

    // Discard small or heavy terms among those changed since the last pass,
    // judged at the bound angles. `cut` is the number of stages before them.
    auto filter = [&](int cut) -> bool {
        for (size_t s = 0; s < terms.capacity(); ++s) {
            if (terms.state[s] != TermTable<W>::CHANGED) continue;
            const double* c = &terms.coefs[s * stride];
            double magnitude = 0.;
            for (int j = 1; j <= ncoefs; ++j) {
                if (!std::isfinite(c[j])) return false;
                magnitude = std::max(magnitude, std::abs(c[j]));
            }
            bool drop = magnitude < rq.truncate_below;
            if (!drop && rq.max_weight >= 0) {
                int weight = 0;
                const uint64_t* k = &terms.keys[s * KW];
                for (int j = 0; j < W; ++j) weight += popcount64(k[j] | k[W + j]);
                drop = weight > rq.max_weight;
            }
            if (!drop) { terms.state[s] = TermTable<W>::LIVE; continue; }
            g.dropped.push_back(static_cast<int32_t>(c[0]) - 1);
            g.dropped_cut.push_back(cut);
            ++g.truncated_terms;
            terms.erase(s);
        }
        return true;
    };
    if (g.truncating && !filter(rq.nstages)) { g.code = -2; return; }
    g.peak_terms = terms.size;

    std::vector<uint64_t> pending_keys;
    std::vector<double> pending;  // node + 1, then coefficients when truncating
    std::vector<uint8_t> pending_rows;
    for (int step = rq.nstages - 1; step >= 0; --step) {
        CStage const& stage = rq.stage(step);
        CGraphSupport const& allowed = support[rq.stage_index(step)];
        int q0, q1;
        rq.wires(step, q0, q1);
        const int dim = stage.arity == 1 ? 4 : 16;
        const bool close0 = layout.first_touch[q0] == step;
        const bool close1 = stage.arity == 2 && layout.first_touch[q1] == step;
        std::array<GraphRow, 16> rows;
        auto build = [&](int row) {
            GraphRow& r = rows[row];
            int count = 0;
            for (int column = 0; column < dim; ++column) {
                if (!((allowed.rows[row] >> column) & 1)) continue;
                if ((close0 && !nonzero_input_overlap(column & 3, rq.initial_mode)) ||
                    (close1 && !nonzero_input_overlap(column >> 2, rq.initial_mode))) continue;
                r.label[count] = static_cast<uint8_t>((close0 ? 0 : (column & 3)) | (close1 ? 0 : (column & 12)));
                r.entry[count++] = static_cast<uint8_t>(row * dim + column);
            }
            r.size = count;
            r.identity = ((allowed.identity >> row) & 1) && count == 1 && r.label[0] == row;
        };
        const int w0 = q0 >> 6, w1 = q1 >> 6;
        const uint64_t b0 = 1ULL << (q0 & 63), b1 = stage.arity == 2 ? 1ULL << (q1 & 63) : 0;

        // Pass 1: identity rows keep their node; every other key leaves.
        pending_keys.clear();
        pending.clear();
        pending_rows.clear();
        for (size_t s = 0; s < terms.capacity(); ++s) {
            const uint8_t st = terms.state[s];
            if (st != TermTable<W>::LIVE && st != TermTable<W>::CHANGED) continue;
            const uint64_t* k = &terms.keys[s * KW];
            int row = ((k[w0] & b0) ? 1 : 0) + ((k[W + w0] & b0) ? 2 : 0);
            if (stage.arity == 2)
                row += (((k[w1] & b1) ? 1 : 0) + ((k[W + w1] & b1) ? 2 : 0)) << 2;
            GraphRow& r = rows[row];
            if (r.size < 0) build(row);
            if (r.identity) continue;
            if (r.size > 0) {
                pending_keys.insert(pending_keys.end(), k, k + KW);
                pending.insert(pending.end(), &terms.coefs[s * stride], &terms.coefs[s * stride] + stride);
                pending_rows.push_back(static_cast<uint8_t>(row));
            }
            terms.erase(s);
        }

        // Pass 2: an edge from each leaving node to each image's node. An
        // image that is already live (an untouched key) accumulates into its
        // node; that node has not been read since it was written.
        const auto begin = static_cast<uint32_t>(g.source.size());
        for (size_t p = 0; p < pending_rows.size(); ++p) {
            GraphRow const& r = rows[pending_rows[p]];
            const double* from = &pending[p * stride];
            for (int i = 0; i < r.size; ++i) {
                const int label = r.label[i];
                std::copy_n(&pending_keys[p * KW], KW, key.begin());
                key[w0] = (key[w0] & ~b0) | ((label & 1) ? b0 : 0);
                key[W + w0] = (key[W + w0] & ~b0) | ((label & 2) ? b0 : 0);
                if (stage.arity == 2) {
                    key[w1] = (key[w1] & ~b1) | ((label & 4) ? b1 : 0);
                    key[W + w1] = (key[W + w1] & ~b1) | ((label & 8) ? b1 : 0);
                }
                const size_t slot = terms.find_or_insert(key.data(), limits.max_terms);
                if (slot == SIZE_MAX) { g.code = -1; return; }
                double* to = &terms.coefs[slot * stride];
                if (to[0] == 0.) {
                    if (g.nodes >= limits.max_nodes) { g.code = GRAPH_NODES; return; }
                    to[0] = ++g.nodes;
                }
                if (g.source.size() >= limits.max_edges) { g.code = GRAPH_EDGES; return; }
                g.source.push_back(static_cast<int32_t>(from[0]) - 1);
                g.destination.push_back(static_cast<int32_t>(to[0]) - 1);
                g.entry.push_back(r.entry[i]);
                terms.state[slot] = TermTable<W>::CHANGED;
                if (g.truncating) {
                    const double t = stage.ptm[r.entry[i]];
                    for (int j = 1; j <= ncoefs; ++j) to[j] += t * from[j];
                    if (stage.param >= 0) to[stage.param + 2] += stage.dptm[r.entry[i]] * from[1];
                }
            }
        }
        const auto end = static_cast<uint32_t>(g.source.size());
        if (end > begin) g.layers.push_back({rq.stage_index(step), begin, end});
        if (g.truncating && !filter(step)) { g.code = -2; return; }
        g.peak_terms = std::max<uint64_t>(g.peak_terms, terms.size);
    }
    // Every wire has been contracted into the input, so only the empty key
    // can remain.
    for (size_t s = 0; s < terms.capacity(); ++s) {
        const uint8_t st = terms.state[s];
        if (st != TermTable<W>::LIVE && st != TermTable<W>::CHANGED) continue;
        const uint64_t* k = &terms.keys[s * KW];
        for (int j = 0; j < KW; ++j)
            if (k[j]) { g.code = GRAPH_BOUNDARY; return; }
        g.outputs.push_back(static_cast<int32_t>(terms.coefs[s * stride]) - 1);
    }
}

// Value, and gradient by one reverse pass. `bound` receives nothing.
template<bool WithGradient>
void evaluate_exact(Graph& g, const CStage* full, double* out) {
    g.values.assign(g.nodes, 0.);
    std::copy(g.initial.begin(), g.initial.end(), g.values.begin());
    double* v = g.values.data();
    const int32_t* src = g.source.data();
    const int32_t* dst = g.destination.data();
    const uint8_t* entry = g.entry.data();
    for (GraphLayer const& layer : g.layers) {
        const double* t = full[layer.stage].ptm;
        for (uint32_t e = layer.begin; e < layer.end; ++e) v[dst[e]] += t[entry[e]] * v[src[e]];
    }
    for (int32_t node : g.outputs) out[0] += v[node];
    if constexpr (WithGradient) {
        g.adjoint.assign(g.nodes, 0.);
        double* a = g.adjoint.data();
        for (int32_t node : g.outputs) a[node] = 1.;
        for (auto layer = g.layers.rbegin(); layer != g.layers.rend(); ++layer) {
            CStage const& stage = full[layer->stage];
            const double* t = stage.ptm;
            if (stage.param >= 0) {
                const double* dt = stage.dptm;
                double tangent = 0.;
                for (uint32_t e = layer->begin; e < layer->end; ++e) {
                    const double pull = a[dst[e]];
                    tangent += pull * v[src[e]] * dt[entry[e]];
                    a[src[e]] += t[entry[e]] * pull;
                }
                out[stage.param + 1] += tangent;
            } else {
                for (uint32_t e = layer->begin; e < layer->end; ++e) a[src[e]] += t[entry[e]] * a[dst[e]];
            }
        }
    }
}

// Truncated value with its bound: sum of the discarded coefficients'
// magnitudes at the bound angles, each <= one in expectation magnitude.
// With gradients, forward tangents give each discarded term's derivative too.
template<bool WithGradient>
void evaluate_truncated(Graph& g, const CStage* full, int nparams, double* out, double* bound) {
    if constexpr (!WithGradient) {
        evaluate_exact<false>(g, full, out);
        for (int32_t node : g.dropped) bound[0] += std::abs(g.values[node]);
        return;
    }
    const int k = nparams + 1;
    g.values.assign(static_cast<size_t>(g.nodes) * k, 0.);
    double* v = g.values.data();
    for (size_t n = 0; n < g.initial.size(); ++n) v[n * k] = g.initial[n];
    for (GraphLayer const& layer : g.layers) {
        CStage const& stage = full[layer.stage];
        const int p = stage.param;
        for (uint32_t e = layer.begin; e < layer.end; ++e) {
            const double* from = &v[static_cast<size_t>(g.source[e]) * k];
            double* to = &v[static_cast<size_t>(g.destination[e]) * k];
            const double t = stage.ptm[g.entry[e]];
            for (int j = 0; j < k; ++j) to[j] += t * from[j];
            if (p >= 0) to[p + 1] += stage.dptm[g.entry[e]] * from[0];
        }
    }
    for (int32_t node : g.outputs)
        for (int j = 0; j < k; ++j) out[j] += v[static_cast<size_t>(node) * k + j];
    for (size_t i = 0; i < g.dropped.size(); ++i) {
        const double* c = &v[static_cast<size_t>(g.dropped[i]) * k];
        const double* before = nparams ? &g.lipschitz_before[static_cast<size_t>(g.dropped_cut[i]) * nparams] : nullptr;
        bound[0] += std::abs(c[0]);
        for (int j = 0; j < nparams; ++j) bound[j + 1] += std::abs(c[j + 1]) + std::abs(c[0]) * before[j];
    }
}

}  // namespace

// Original public C symbols and structures remain ABI compatible.
PAULI_API int run_pauli(
    int nqubits, int nparams, int nstages, const CStage* stages,
    int nterms, const CTerm* observables, int initial_mode,
    uint64_t max_terms, double* out, CStats* stats
) {
    return run_single<true>(nqubits, nparams, nstages, stages, nstages, nullptr,
                            nterms, observables, initial_mode, max_terms, out, stats);
}
PAULI_API int run_pauli_indexed(
    int nqubits, int nparams, int nstages, const CStage* full_stages,
    int nfull, const int32_t* indices,
    int nterms, const CTerm* observables, int initial_mode,
    uint64_t max_terms, double* out, CStats* stats
) {
    if (nstages > 0 && !indices) return -2;
    return run_single<true>(nqubits, nparams, nstages, full_stages, nfull, indices,
                            nterms, observables, initial_mode, max_terms, out, stats);
}

// Optional value-only symbols use the original parameter count for validation,
// but write exactly one output double and never read derivative matrices.
PAULI_API int run_pauli_value(
    int nqubits, int nparams, int nstages, const CStage* stages,
    int nterms, const CTerm* observables, int initial_mode,
    uint64_t max_terms, double* out, CStats* stats
) {
    return run_single<false>(nqubits, nparams, nstages, stages, nstages, nullptr,
                             nterms, observables, initial_mode, max_terms, out, stats);
}
PAULI_API int run_pauli_indexed_value(
    int nqubits, int nparams, int nstages, const CStage* full_stages,
    int nfull, const int32_t* indices,
    int nterms, const CTerm* observables, int initial_mode,
    uint64_t max_terms, double* out, CStats* stats
) {
    if (nstages > 0 && !indices) return -2;
    return run_single<false>(nqubits, nparams, nstages, full_stages, nfull, indices,
                             nterms, observables, initial_mode, max_terms, out, stats);
}

// All cones of one evaluation in one call. Groups run on up to `nthreads`
// threads; results are summed in group order, so the output does not depend
// on the thread count. On failure, failed_group is the lowest failing group.
// A group over the term cap does not stop the others, so one call reports
// every such group in group_codes (optional, one code per group) and the
// caller can split them all before retrying.
PAULI_API int run_pauli_batch(
    int nparams, const CStage* full_stages, int nfull, const double* lipschitz,
    int ngroups, const CGroup* groups, int initial_mode, uint64_t max_terms,
    double truncate_below, int32_t max_weight, int with_gradient, int nthreads,
    double* out, CBatchStats* stats, int32_t* group_codes
) {
    if (ngroups < 0 || (ngroups && !groups) || !out || !stats || nthreads < 1 ||
        nparams < 0 || nparams > 16) return -2;
    std::vector<Result> results;
    try { results.resize(ngroups); } catch (std::bad_alloc const&) { return -3; }
    auto run = [&](int g) {
        CGroup const& group = groups[g];
        Request rq{group.nqubits, nparams, group.nstages, full_stages, nfull, group.indices,
                   group.local_wires, group.nterms, group.terms, initial_mode, max_terms,
                   truncate_below, max_weight, lipschitz};
        if (group.nstages > 0 && !group.indices) { results[g].code = -2; return; }
        results[g] = with_gradient ? run_request<true>(rq) : run_request<false>(rq);
    };
    const int pool = parallel_for(ngroups, nthreads, [&](int g) {
        run(g);
        return results[g].code == 0 || results[g].code == -1;
    });
    if (pool) return pool;
    if (group_codes)
        for (int g = 0; g < ngroups; ++g) group_codes[g] = results[g].code;
    const int noutputs = with_gradient ? nparams + 1 : 1;
    std::fill_n(out, noutputs, 0.);
    *stats = CBatchStats{};
    stats->failed_group = -1;
    for (int g = 0; g < ngroups; ++g) {
        Result const& r = results[g];
        if (r.code) {
            stats->failed_group = g;
            return r.code;
        }
        for (int j = 0; j < noutputs; ++j) out[j] += r.out[j];
        stats->peak_terms = std::max(stats->peak_terms, r.peak_terms);
        stats->final_terms += r.final_terms;
        stats->truncated_terms += r.truncated_terms;
        stats->value_bound += r.value_bound;
        for (int j = 0; j < nparams; ++j) stats->gradient_bound[j] += r.gradient_bound[j];
    }
    return 0;
}

// Compile one structural graph per cone group. On failure *handle stays null
// and stats->failed_group is the lowest failing group; node and edge budgets
// apply to the total over groups, accumulated in group order. As in
// run_pauli_batch, a group over the term cap does not stop the others, and
// group_codes (optional) receives every group's code.
PAULI_API int pauli_graph_compile(
    int nparams, const CStage* full_stages, int nfull, const CGraphSupport* support,
    int ngroups, const CGroup* groups, int initial_mode, uint64_t max_terms,
    uint64_t max_nodes, uint64_t max_edges, double truncate_below, int32_t max_weight,
    const double* lipschitz, int nthreads, void** handle, CGraphStats* stats, int32_t* group_codes
) {
    if (!handle || !stats) return -2;
    *handle = nullptr;
    *stats = CGraphStats{};
    stats->failed_group = -1;
    if (ngroups < 0 || (ngroups && !groups) || nthreads < 1 || nparams < 0 || nparams > GRAPH_MAX_PARAMS ||
        nfull < 0 || (nfull && (!full_stages || !support)) || max_terms < 1 ||
        max_nodes < 1 || max_nodes > UINT32_MAX - 1 || max_edges < 1 || max_edges > UINT32_MAX ||
        !(truncate_below >= 0.) || !std::isfinite(truncate_below) || max_weight < -1 ||
        (initial_mode != 0 && initial_mode != 1)) return -2;
    GraphSet* set = nullptr;
    try {
        set = new GraphSet{nfull, std::vector<Graph>(ngroups)};
    } catch (std::bad_alloc const&) { return -3; }
    const GraphLimits limits{max_terms, max_nodes, max_edges};
    const int pool = parallel_for(ngroups, nthreads, [&](int i) {
        CGroup const& group = groups[i];
        Graph& g = set->groups[i];
        Request rq{group.nqubits, nparams, group.nstages, full_stages, nfull, group.indices,
                   group.local_wires, group.nterms, group.terms, initial_mode, max_terms,
                   truncate_below, max_weight, lipschitz};
        if (group.nqubits < 1 || group.nqubits > 256 || group.nstages < 0 || group.nterms < 0 ||
            (group.nstages && !group.indices) || (group.nterms && !group.terms)) {
            g.code = -2;
            return false;
        }
        try {
            const int words = (group.nqubits + 63) / 64;
            if (words == 1) compile_graph<1>(rq, support, limits, g);
            else if (words == 2) compile_graph<2>(rq, support, limits, g);
            else if (words == 3) compile_graph<3>(rq, support, limits, g);
            else compile_graph<4>(rq, support, limits, g);
        } catch (std::bad_alloc const&) { g.code = -3; }
          catch (...) { g.code = -4; }
        return g.code == 0 || g.code == -1;
    });
    if (group_codes)
        for (int i = 0; i < ngroups; ++i) group_codes[i] = set->groups[i].code;
    int code = pool;
    for (int i = 0; i < ngroups && !code; ++i) {
        Graph const& g = set->groups[i];
        code = g.code;
        if (!code) {
            stats->nodes += g.nodes;
            stats->edges += g.source.size();
            stats->peak_terms = std::max(stats->peak_terms, g.peak_terms);
            stats->bytes += g.bytes();
            stats->truncated_terms += g.truncated_terms;
            if (stats->nodes > max_nodes) code = GRAPH_NODES;
            else if (stats->edges > max_edges) code = GRAPH_EDGES;
        }
        if (code) stats->failed_group = i;
    }
    if (code) {
        delete set;
        return code;
    }
    *handle = set;
    return 0;
}

// Evaluate every group's graph against the bound stages and sum the results
// in group order, so the output does not depend on the thread count. For a
// truncated graph, `bounds` (1 + nparams doubles, value first) receives the
// rigorous error bounds of the value and, with gradients, of each derivative.
PAULI_API int pauli_graph_evaluate(
    void* handle, int nparams, const CStage* full_stages, int nfull, int with_gradient,
    int nthreads, double* out, double* bounds
) {
    auto* set = static_cast<GraphSet*>(handle);
    if (!set || !out || nthreads < 1 || nparams < 0 || nparams > GRAPH_MAX_PARAMS || nfull != set->nfull ||
        (nfull && !full_stages)) return -2;
    for (int i = 0; i < nfull; ++i)
        if (full_stages[i].param < -1 || full_stages[i].param >= nparams) return -2;
    const int noutputs = with_gradient ? nparams + 1 : 1;
    // One row of nparams + 1 outputs per group, summed afterwards in group order.
    const size_t width = static_cast<size_t>(nparams) + 1;
    std::vector<double> partial, partial_bounds;
    try {
        partial.assign(set->groups.size() * width, 0.);
        partial_bounds.assign(set->groups.size() * width, 0.);
    } catch (std::bad_alloc const&) { return -3; }
    // A thread pays off only with enough edges to keep it busy for longer than
    // starting it takes (tens of microseconds).
    uint64_t edges = 0;
    for (Graph const& g : set->groups) edges += g.source.size();
    const int workers = static_cast<int>(std::min<uint64_t>(nthreads, std::max<uint64_t>(1, edges / 200000)));
    std::atomic<int> code{0};
    const int pool = parallel_for(static_cast<int>(set->groups.size()), workers, [&](int i) {
        Graph& g = set->groups[i];
        double* o = &partial[static_cast<size_t>(i) * width];
        double* b = &partial_bounds[static_cast<size_t>(i) * width];
        try {
            if (g.nodes == 0) return true;
            if (g.truncating) {
                if (with_gradient) evaluate_truncated<true>(g, full_stages, nparams, o, b);
                else evaluate_truncated<false>(g, full_stages, nparams, o, b);
            } else if (with_gradient) evaluate_exact<true>(g, full_stages, o);
            else evaluate_exact<false>(g, full_stages, o);
        } catch (std::bad_alloc const&) { code = -3; return false; }
        return true;
    });
    if (pool) return pool;
    if (code) return code;
    std::fill_n(out, noutputs, 0.);
    for (size_t i = 0; i < set->groups.size(); ++i)
        for (int j = 0; j < noutputs; ++j) out[j] += partial[i * width + j];
    for (int j = 0; j < noutputs; ++j)
        if (!std::isfinite(out[j])) return -5;
    if (bounds) {
        std::fill_n(bounds, noutputs, 0.);
        for (size_t i = 0; i < set->groups.size(); ++i)
            for (int j = 0; j < noutputs; ++j) bounds[j] += partial_bounds[i * width + j];
        for (int j = 0; j < noutputs; ++j)
            if (!std::isfinite(bounds[j])) return -5;
    }
    return 0;
}

PAULI_API void pauli_graph_free(void* handle) {
    delete static_cast<GraphSet*>(handle);
}

// Rebind every parameterised single-wire rotation for a new shift vector.
// Rotation r about the axis that mixes Pauli labels first[r] and second[r]
// has angle base + scale * shifts[param]; it writes four entries of its
// 4x4 transfer matrix and of their derivatives with respect to the shift.
PAULI_API void pauli_bind_rotations(CStage* stages, int nrotations, const int32_t* stage, const int32_t* first,
                                    const int32_t* second, const int32_t* param, const double* base,
                                    const double* scale, const double* sign, const double* shifts) {
    for (int r = 0; r < nrotations; ++r) {
        const double theta = base[r] + scale[r] * shifts[param[r]];
        const double c = std::cos(theta), s = std::sin(theta);
        const int a = first[r], b = second[r];
        double* t = stages[stage[r]].ptm;
        double* d = stages[stage[r]].dptm;
        t[a * 5] = c;
        t[b * 5] = c;
        t[a * 4 + b] = sign[r] * s;
        t[b * 4 + a] = -sign[r] * s;
        d[a * 5] = -scale[r] * s;
        d[b * 5] = -scale[r] * s;
        d[a * 4 + b] = sign[r] * scale[r] * c;
        d[b * 4 + a] = -sign[r] * scale[r] * c;
    }
}
