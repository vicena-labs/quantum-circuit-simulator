# Scoreboard: AMD EPYC 7502P 32-Core Processor, 16 logical CPUs, 32.0 GiB

16 thread(s) per tool; Linux-5.15.0-102-generic-x86_64-with-glibc2.36; Python 3.12.10. Versions: quantum-circuit-simulator 0.5.0rc5, qiskit 2.6.2, qiskit-aer 0.18.4, qulacs 0.6.14, pennylane 0.45.1, pennylane-lightning 0.45.0, quimb 1.15.0, pauli-prop 0.2.1, spd git f694c9c.

Each figure is the median of 5 fresh parameter vectors in one fresh process with a time limit. Other tools run as their authors recommend: PauliPropagation.jl with `VectorPauliSum`, its `Performance` kernels and `min_abs_coeff = 1e-12`; pauli-prop with `atol = 1e-12`; sparse Pauli dynamics (spd) with its Clifford parts moved into the Pauli frame once, commuting rotations merged and threshold 1e-12. Every other tool also runs on circuits cut to the observable's light cone, as one circuit (cone) or one per observable term (split), and counts its fastest setup. Quantum Circuit Simulator (QCS) runs uncut circuits, exactly and at the same 1e-12 threshold. Ratios below 10× are in bold. *Repeated* is one more evaluation of a prepared circuit, as in an optimiser or a sweep; *single* adds the setup of the circuit, QCS graph compilation included and once-per-process costs such as imports and Julia's compilation excluded.

## Summary

- **Repeated:** 14 workloads at 10× or more, 2 below, 3 that only QCS finished, 4 that QCS refused and another tool finished, 3 that no tool finished.
- **Single:** 4 workloads at 10× or more, 12 below, 3 that only QCS finished, 4 that QCS refused and another tool finished, 3 that no tool finished.

## Values

| Workload | QCS repeated (backend, mode, error) | Fastest other, repeated (setup, error) | QCS lead | QCS single | Fastest other, single | QCS lead |
|---|---|---|---|---|---|---|
| A_vqe_3x4_L2 | 21.1 µs (graph, exact, 9e-16) | PauliPropagation.jl 158 µs (whole, 3e-15) | **7.49×** | 2.11 ms | Qulacs 1.07 ms | **1.98× slower** |
| A_vqe_4x5_L2 | 22.5 µs (graph, exact, 4e-15) | PauliPropagation.jl 239 µs (cone, 5e-15) | 10.6× | 3.4 ms | Qulacs 5.99 ms | **1.76×** |
| A_vqe_4x5_L4 | 453 µs (graph, exact, 0e+00) | Aer 42 ms (cone, 2e-14) | 92.8× | 31.7 ms | Aer 47.3 ms | **1.49×** |
| A_vqe_4x6_L6 | refused | Aer 968 ms (cone, no ref) | — | — | Aer 974 ms | — |
| B_noisy_2x5_L4 | 26.2 µs (graph, exact, 1e-16) | PauliPropagation.jl 494 µs (whole, 2e-16) | 18.8× | 3 ms | PauliPropagation.jl 38.5 ms | 12.8× |
| B_noisy_6x6_L3 | 27.5 µs (graph, exact, 3e-16) | PauliPropagation.jl 481 µs (cone, 4e-16) | 17.5× | 7.94 ms | PauliPropagation.jl 40 ms | **5.04×** |
| B_noisy_6x6_L5 | 631 µs (graph, exact, 2e-16) | PauliPropagation.jl 5.17 s (cone, 1e-11) | 8.19e+03× | 37.3 ms | PauliPropagation.jl 5.24 s | 141× |
| B_noisy_20x20_L3 | 146 µs (graph, exact, 0e+00) | PauliPropagation.jl 28.9 ms (split, 4e-16) | 198× | 111 ms | PauliPropagation.jl 123 ms | **1.1×** |
| B_noisy_32x32_L3 | 318 µs (graph, exact, 1e-16) | PauliPropagation.jl 86.8 ms (split, 6e-16) | 273× | 294 ms | PauliPropagation.jl 227 ms | **1.3× slower** |
| C_centre_10x10_L4 | 44.7 µs (graph, exact, 3e-17) | PauliPropagation.jl 213 µs (split, 0e+00) | **4.76×** | 19.6 ms | Qulacs 3.18 ms | **6.15× slower** |
| C_centre_10x10_L6 | refused | quimb 181 ms (split, no ref) | — | — | quimb 182 ms | — |
| C_centre_10x10_L8 | refused | quimb 500 ms (split, no ref) | — | — | quimb 501 ms | — |
| C_centre_32x32_L6 | refused | quimb 394 ms (cone, no ref) | — | — | quimb 403 ms | — |
| D_noisy1q_2x5_L4 | 26.4 µs (graph, exact, 1e-16) | PauliPropagation.jl 446 µs (cone, 4e-16) | 16.9× | 3.42 ms | PauliPropagation.jl 38.4 ms | 11.2× |
| D_noisy1q_6x6_L3 | 29.5 µs (graph, exact, 3e-16) | PauliPropagation.jl 752 µs (cone, 1e-16) | 25.4× | 9.42 ms | PauliPropagation.jl 39.9 ms | **4.23×** |
| D_noisy1q_6x6_L5 | 636 µs (graph, exact, 0e+00) | PauliPropagation.jl 5.22 s (whole, 1e-11) | 8.21e+03× | 39.8 ms | PauliPropagation.jl 5.27 s | 132× |
| D_noisy1q_20x20_L3 | 145 µs (graph, exact, 2e-16) | PauliPropagation.jl 36 ms (split, 4e-16) | 249× | 134 ms | PauliPropagation.jl 126 ms | **1.07× slower** |
| D_noisy1q_32x32_L3 | 346 µs (graph, exact, 0e+00) | PauliPropagation.jl 80.8 ms (split, 4e-16) | 234× | 342 ms | PauliPropagation.jl 236 ms | **1.45× slower** |
| D_noisy1q_6x6_L6 | refused | none finished | — | — | — | — |
| D_noisy1q_10x10_L6 | refused | none finished | — | — | — | — |
| K_eagle_mz_S5_t0pi8 | 269 µs (graph, 1e-12, 0e+00) | PauliPropagation.jl 11.7 ms (cone, 0e+00) | 43.5× | 61.4 ms | PauliPropagation.jl 57.3 ms | **1.07× slower** |
| K_eagle_mz_S5_t1pi8 | 81.4 ms (native, exact, 0e+00) | none finished | — | 140 ms | — | — |
| K_eagle_mz_S5_t2pi8 | 81.3 ms (native, exact, 0e+00) | none finished | — | 140 ms | — | — |
| K_eagle_mz_S5_t3pi8 | 82.5 ms (native, 1e-12, 0e+00) | none finished | — | 140 ms | — | — |
| K_eagle_mz_S5_t4pi8 | 7.02 ms (graph, 1e-12, 0e+00) | PauliPropagation.jl 160 ms (cone, 0e+00) | 22.8× | 70.7 ms | PauliPropagation.jl 205 ms | **2.9×** |
| K_eagle_z62_S20_t1pi8 | refused | none finished | — | — | — | — |

## Gradients of every shared parameter, repeated

No other tool returns gradients through noise channels; those rows show QCS alone.

| Workload | QCS (backend, mode) | Fastest other (setup) | QCS lead |
|---|---|---|---|
| A_vqe_3x4_L2 | 38 µs (graph, exact) | PauliPropagation.jl 900 µs (cone) | 23.7× |
| A_vqe_4x5_L2 | 41.5 µs (graph, exact) | PauliPropagation.jl 2.03 ms (whole) | 48.8× |
| A_vqe_4x5_L4 | 1.02 ms (graph, exact) | Qulacs 681 ms (whole) | 668× |
| B_noisy_2x5_L4 | 48 µs (graph, exact) | none | — |
| B_noisy_6x6_L3 | 49.6 µs (graph, exact) | none | — |
| B_noisy_6x6_L5 | 1.4 ms (graph, exact) | none | — |
| B_noisy_20x20_L3 | 207 µs (graph, exact) | none | — |
| B_noisy_32x32_L3 | 437 µs (graph, exact) | none | — |
| C_centre_10x10_L4 | 62.7 µs (graph, exact) | PauliPropagation.jl 2.43 ms (cone) | 38.8× |
| D_noisy1q_2x5_L4 | 53.2 µs (graph, exact) | none | — |
| D_noisy1q_6x6_L3 | 48.4 µs (graph, exact) | none | — |
| D_noisy1q_6x6_L5 | 1.47 ms (graph, exact) | none | — |
| D_noisy1q_20x20_L3 | 212 µs (graph, exact) | none | — |
| D_noisy1q_32x32_L3 | 492 µs (graph, exact) | none | — |
| K_eagle_mz_S5_t0pi8 | 615 µs (graph, 1e-12) | PauliPropagation.jl 87.6 ms (cone) | 142× |
| K_eagle_mz_S5_t1pi8 | 92.8 ms (native, 1e-12) | none | — |
| K_eagle_mz_S5_t2pi8 | 91.5 ms (native, exact) | none | — |
| K_eagle_mz_S5_t3pi8 | 91.8 ms (native, exact) | none | — |
| K_eagle_mz_S5_t4pi8 | 22 ms (graph, 1e-12) | PauliPropagation.jl 526 ms (split) | 23.9× |

## Every configuration, repeated evaluation

| Workload | QCS graph 1e-12 | QCS graph exact | QCS native 1e-12 | QCS native exact | Aer cone | Aer split | Aer whole | Lightning cone | Lightning split | Lightning whole | PauliPropagation.jl cone | PauliPropagation.jl split | PauliPropagation.jl whole | Qulacs cone | Qulacs split | Qulacs whole | pauli-prop cone | pauli-prop split | pauli-prop whole | quimb cone | quimb split | quimb whole | spd cone | spd split | spd whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_vqe_3x4_L2 | 44.1 µs | 21.1 µs | 204 µs | 129 µs | 4.76 ms | 25.3 ms | 3.25 ms | 6.83 ms | 111 ms | 7 ms | 398 µs | 2.08 ms | 158 µs | 854 µs | 851 µs | 874 µs | 3.96 ms | 31.9 ms | 3.84 ms | 289 ms | 367 ms | 298 ms | 6.31 ms | 30.1 ms | 5.82 ms |
| A_vqe_4x5_L2 | 39.5 µs | 22.5 µs | 208 µs | 189 µs | 83.7 ms | 41.7 ms | 29.6 ms | 282 ms | 181 ms | 237 ms | 239 µs | 2.94 ms | 345 µs | 99.7 ms | 1.68 ms | 68.7 ms | 7 ms | 56.3 ms | 7.11 ms | 599 ms | 707 ms | 587 ms | 10.2 ms | 51.1 ms | 10.4 ms |
| A_vqe_4x5_L4 | 464 µs | 453 µs | 25.8 ms | 25.8 ms | 42 ms | 158 ms | 45.6 ms | 328 ms | 745 ms | 330 ms | 798 ms | timeout | 781 ms | 114 ms | 113 ms | 112 ms | timeout | timeout | timeout | 6.76 s | 7.19 s | 6.81 s | timeout | timeout | timeout |
| A_vqe_4x6_L6 | refused | refused | refused | refused | 968 ms | 10.1 s | 975 ms | timeout | failed | timeout | failed | failed | failed | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| B_noisy_2x5_L4 | 46.6 µs | 26.2 µs | 292 µs | 284 µs | 68.3 ms | 47.4 ms | 68.3 ms | — | — | — | 571 µs | 1.47 ms | 494 µs | 1.17 s | 129 ms | 723 ms | 15.8 ms | 49.4 ms | 15.6 ms | — | — | — | — | — | — |
| B_noisy_6x6_L3 | 45 µs | 27.5 µs | 345 µs | 339 µs | — | 157 ms | — | — | — | — | 481 µs | 3.3 ms | 889 µs | — | 715 ms | — | 12.5 ms | 47.3 ms | 12 ms | — | — | — | — | — | — |
| B_noisy_6x6_L5 | 650 µs | 631 µs | 24.2 ms | 24.3 ms | — | — | — | — | — | — | 5.17 s | 6.36 s | 5.26 s | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| B_noisy_20x20_L3 | — | 146 µs | 1.77 ms | 1.86 ms | — | 3.16 s | — | — | — | — | 76.2 ms | 28.9 ms | 77.4 ms | — | 11.2 s | — | 1.02 s | 558 ms | 1.03 s | — | — | — | — | — | — |
| B_noisy_32x32_L3 | — | 318 µs | 2.97 ms | 2.99 ms | — | 8.4 s | — | — | — | — | 1.09 s | 86.8 ms | 1.09 s | — | timeout | — | timeout | 1.43 s | timeout | — | — | — | — | — | — |
| C_centre_10x10_L4 | 60.6 µs | 44.7 µs | 248 µs | 249 µs | 5.12 ms | 3.96 ms | — | 8.66 ms | 8.57 ms | — | 234 µs | 213 µs | 1.45 ms | 2.56 ms | 2.58 ms | — | 10.2 ms | 10.2 ms | 42.5 ms | 53.6 ms | 53.8 ms | 230 ms | 6.24 ms | 6.21 ms | 45.2 ms |
| C_centre_10x10_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | 187 ms | 181 ms | 545 ms | timeout | timeout | timeout |
| C_centre_10x10_L8 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | 1e+03 ms | 500 ms | 1.08 s | timeout | timeout | timeout |
| C_centre_32x32_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | timeout | — | — | — | timeout | timeout | timeout | 394 ms | 496 ms | 4.72 s | timeout | timeout | timeout |
| D_noisy1q_2x5_L4 | 50.8 µs | 26.4 µs | 309 µs | 297 µs | 66.1 ms | 44.2 ms | 66.1 ms | — | — | — | 446 µs | 1.9 ms | 515 µs | 410 ms | 141 ms | 408 ms | 15.2 ms | 48.7 ms | 15.3 ms | — | — | — | — | — | — |
| D_noisy1q_6x6_L3 | 50.9 µs | 29.5 µs | 389 µs | 386 µs | — | 255 ms | — | — | — | — | 752 µs | 2.79 ms | 1.31 ms | — | 720 ms | — | 11.1 ms | 47.4 ms | 11.2 ms | — | — | — | — | — | — |
| D_noisy1q_6x6_L5 | 663 µs | 636 µs | 24.7 ms | 24.5 ms | — | — | — | — | — | — | 5.34 s | 6.28 s | 5.22 s | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| D_noisy1q_20x20_L3 | — | 145 µs | 1.94 ms | 1.98 ms | — | 3.11 s | — | — | — | — | 77.3 ms | 36 ms | 80.1 ms | — | 11 s | — | 929 ms | 556 ms | 914 ms | — | — | — | — | — | — |
| D_noisy1q_32x32_L3 | — | 346 µs | 3.02 ms | 3.12 ms | — | 8.29 s | — | — | — | — | 1.06 s | 80.8 ms | 1.12 s | — | timeout | — | timeout | 1.43 s | timeout | — | — | — | — | — | — |
| D_noisy1q_6x6_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| D_noisy1q_10x10_L6 | refused | refused | refused | refused | — | — | — | — | — | — | timeout | failed | timeout | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| K_eagle_mz_S5_t0pi8 | 269 µs | — | 4.72 ms | 7.98 ms | — | — | — | — | — | — | 11.7 ms | 85.1 ms | 16.5 ms | — | — | — | 92.6 ms | 338 ms | 92.8 ms | timeout | timeout | timeout | 140 ms | timeout | 146 ms |
| K_eagle_mz_S5_t1pi8 | — | — | 81.4 ms | 81.4 ms | — | — | — | — | — | — | timeout | timeout | timeout | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t2pi8 | — | — | 81.5 ms | 81.3 ms | — | — | — | — | — | — | failed | timeout | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t3pi8 | — | — | 82.5 ms | 82.6 ms | — | — | — | — | — | — | failed | timeout | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t4pi8 | 7.02 ms | — | 12.4 ms | 26 ms | — | — | — | — | — | — | 160 ms | 207 ms | 165 ms | — | — | — | 3.45 s | 2.7 s | 3.43 s | timeout | timeout | timeout | 2.45 s | timeout | 2.49 s |
| K_eagle_z62_S20_t1pi8 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
