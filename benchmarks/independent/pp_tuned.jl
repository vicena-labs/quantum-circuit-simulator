# PauliPropagation.jl run the way its authors recommend (in
# the package's advanced_performance example): the
# VectorPauliSum type, the fused kernels of the Performance sub-module, a
# coefficient threshold just above rounding noise, and circuits cut to the
# observable's light cone on the smallest system the cone fits in.
#
# usage: julia --project=<env> -t <threads> pp_tuned.jl <workloads.json> <workload> <parts> <min_abs_coeff>
# <parts> is `whole` (the full circuit), `cone` (one circuit for the union of
# every term's light cone) or `split` (one circuit per observable term, summed);
# bench.py export writes all three.
using PauliPropagation, JSON3, Statistics
using PauliPropagation.Performance
const PB = PauliPropagation.PropagationBase

const t_start = time()
data = JSON3.read(read(ARGS[1], String))
w = data[Symbol(ARGS[2])]
mode = ARGS[3]
tol = parse(Float64, ARGS[4])

# Two-wire depolarizing, written as a custom gate: PauliPropagation.jl ships
# only one-wire Pauli channels. It damps every Pauli that is non-identity on
# either wire by 1 - lambda and never branches, like the built-in PauliNoise.
struct TwoQubitDepolarizing <: PauliPropagation.ParametrizedNoiseChannel
    q1::Int
    q2::Int
end

function PB.applytoall!(g::TwoQubitDepolarizing, cache::PauliPropagation.AbstractPauliPropagationCache, lambda;
                        thread::Bool=true, kwargs...)
    d = 1 - lambda
    damp(pstr, c) = (getpauli(pstr, g.q1) != 0 || getpauli(pstr, g.q2) != 0) ? c * d : c
    return mapcoeffsbypair!(damp, cache; thread)
end
PB.requiresmerging(::TwoQubitDepolarizing, ::PB.AbstractPropagationCache) = false

function build(p)
    n = Int(p.n)
    circ = Any[]
    rot = Tuple{Float64,Int}[]          # (base angle, shared parameter) per rotation, in order
    for op in p.ops
        k = String(op[1])
        if k in ("RX", "RY", "RZ")
            push!(circ, PauliRotation(Symbol(k[2:2]), Int(op[2]) + 1))
            push!(rot, (Float64(op[3]), Int(op[4])))
        elseif k == "CZ"
            push!(circ, CliffordGate(:CZ, [Int(op[2]) + 1, Int(op[3]) + 1]))
        elseif k == "depol"
            # Quantum Circuit Simulator's p is the total non-identity probability:
            # one wire damps by 1 - 4p/3, two wires by 1 - 16p/15.
            qs, pr = op[2], Float64(op[3])
            push!(circ, length(qs) == 1 ? DepolarizingNoise(Int(qs[1]) + 1, 4pr / 3) :
                        FrozenGate(TwoQubitDepolarizing(Int(qs[1]) + 1, Int(qs[2]) + 1), 16pr / 15))
        else
            error("unsupported op $k")
        end
    end
    psum = PauliSum(n)
    for (c, o) in p.obs
        qs = [parse(Int, String(q)) + 1 for q in keys(o)]
        add!(psum, [Symbol(String(o[q])) for q in keys(o)], qs, Float64(c))
    end
    return [g for g in circ], rot, VectorPauliSum(psum)
end

thetas(rot, s) = [b + (par >= 0 ? s[par + 1] : 0.0) for (b, par) in rot]

function value(parts, s)
    v, nterms = 0.0, 0
    for (circ, rot, psum) in parts
        out = Performance.propagate(circ, psum, thetas(rot, s); min_abs_coeff=tol)
        v += overlapwithzero(out); nterms += length(out)
    end
    return v, nterms
end

function gradient(parts, s, nparams)
    e, shared = 0.0, zeros(nparams)
    for (circ, rot, psum) in parts
        pe, g = rewindgradient(circ, psum, thetas(rot, s), overlapwithzero; min_abs_coeff=tol)
        e += pe
        for (i, (_, par)) in enumerate(rot)
            par >= 0 && (shared[par + 1] += g[i])
        end
    end
    return e, shared
end

rec = Dict{String,Any}("workload" => ARGS[2], "tool" => "PauliPropagation.jl tuned", "parts_mode" => mode,
                       "threads" => Threads.nthreads(), "min_abs_coeff" => tol,
                       "package_version" => string(pkgversion(PauliPropagation)))
try
    t = time()
    raw = mode == "whole" ? [w] : w.parts[Symbol(mode)]
    parts = [build(p) for p in raw]
    rec["build_s"] = time() - t
    rec["parts"] = length(parts); rec["max_part_qubits"] = maximum(Int(p.n) for p in raw)
    vecs = [Float64.(v) for v in w.vecs]
    reps = (length(vecs) - 1) ÷ 2
    t = time(); v, nterms = value(parts, vecs[1]); rec["first_value_s"] = time() - t
    rec["value"] = v; rec["final_terms"] = nterms
    rec["since_start_to_first_value_s"] = time() - t_start
    ts = Float64[]
    for s in vecs[2:1 + reps]
        t = time(); value(parts, s); push!(ts, time() - t)
    end
    rec["warm_value_s"] = median(ts); rec["warm_value_samples"] = ts
    # Building again, now compiled, times the setup of one more circuit.
    t = time(); [build(p) for p in raw]; rec["rebuild_s"] = time() - t
    if !w.noisy
        t = time(); e, g = gradient(parts, vecs[1], Int(w.nparams)); rec["first_grad_s"] = time() - t
        rec["gradient"] = g; rec["grad_value"] = e
        ts = Float64[]
        for s in vecs[2 + reps:end]
            t = time(); gradient(parts, s, Int(w.nparams)); push!(ts, time() - t)
        end
        rec["warm_grad_s"] = median(ts); rec["warm_grad_samples"] = ts
    else
        rec["grad_unsupported"] = "rewindgradient does not support noise channels"
    end
catch e
    rec["error"] = sprint(showerror, e)[1:min(end, 300)]
end
rec["peak_rss_mb"] = Sys.maxrss() / 1e6
println("RESULT ", JSON3.write(rec))
