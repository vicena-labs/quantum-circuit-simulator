# Scoreboard: AMD EPYC 7502P 32-Core Processor, 16 logical CPUs, 32.0 GiB

1 thread(s) per tool; Linux-5.15.0-102-generic-x86_64-with-glibc2.36; Python 3.12.10. Versions: quantum-circuit-simulator 0.5.0rc5, qiskit 2.6.2, qiskit-aer 0.18.4, qulacs 0.6.14, pennylane 0.45.1, pennylane-lightning 0.45.0, quimb 1.15.0, pauli-prop 0.2.1, spd git f694c9c.

Each figure is the median of 5 fresh parameter vectors in one fresh process with a time limit. Other tools run as their authors recommend: PauliPropagation.jl with `VectorPauliSum`, its `Performance` kernels and `min_abs_coeff = 1e-12`; pauli-prop with `atol = 1e-12`; sparse Pauli dynamics (spd) with its Clifford parts moved into the Pauli frame once, commuting rotations merged and threshold 1e-12. Every other tool also runs on circuits cut to the observable's light cone, as one circuit (cone) or one per observable term (split), and counts its fastest setup. Quantum Circuit Simulator (QCS) runs uncut circuits, exactly and at the same 1e-12 threshold. Ratios below 10× are in bold. *Repeated* is one more evaluation of a prepared circuit, as in an optimiser or a sweep; *single* adds the setup of the circuit, QCS graph compilation included and once-per-process costs such as imports and Julia's compilation excluded.

## Summary

- **Repeated:** 14 workloads at 10× or more, 2 below, 3 that only QCS finished, 4 that QCS refused and another tool finished, 3 that no tool finished.
- **Single:** 5 workloads at 10× or more, 11 below, 3 that only QCS finished, 4 that QCS refused and another tool finished, 3 that no tool finished.

## Values

| Workload | QCS repeated (backend, mode, error) | Fastest other, repeated (setup, error) | QCS lead | QCS single | Fastest other, single | QCS lead |
|---|---|---|---|---|---|---|
| A_vqe_3x4_L2 | 19.4 µs (graph, exact, 9e-16) | PauliPropagation.jl 116 µs (whole, 3e-15) | **5.98×** | 2.18 ms | Qulacs 1.72 ms | **1.27× slower** |
| A_vqe_4x5_L2 | 20.9 µs (graph, exact, 4e-15) | PauliPropagation.jl 221 µs (cone, 5e-15) | 10.6× | 3.5 ms | Qulacs 5.52 ms | **1.58×** |
| A_vqe_4x5_L4 | 472 µs (graph, 1e-12, 0e+00) | Lightning 422 ms (cone, 5e-14) | 895× | 31.6 ms | Lightning 426 ms | 13.5× |
| A_vqe_4x6_L6 | refused | Aer 14 s (cone, no ref) | — | — | Aer 14 s | — |
| B_noisy_2x5_L4 | 27.8 µs (graph, exact, 1e-16) | PauliPropagation.jl 379 µs (whole, 2e-16) | 13.6× | 2.99 ms | PauliPropagation.jl 38.6 ms | 12.9× |
| B_noisy_6x6_L3 | 27.1 µs (graph, exact, 3e-16) | PauliPropagation.jl 461 µs (cone, 4e-16) | 17× | 7.94 ms | PauliPropagation.jl 41.4 ms | **5.22×** |
| B_noisy_6x6_L5 | 631 µs (graph, exact, 2e-16) | PauliPropagation.jl 13.9 s (cone, 1e-11) | 2.2e+04× | 37.7 ms | PauliPropagation.jl 14 s | 370× |
| B_noisy_20x20_L3 | 146 µs (graph, exact, 0e+00) | PauliPropagation.jl 14.8 ms (split, 4e-16) | 101× | 116 ms | PauliPropagation.jl 115 ms | **1.01× slower** |
| B_noisy_32x32_L3 | 321 µs (graph, exact, 1e-16) | PauliPropagation.jl 39.1 ms (split, 6e-16) | 122× | 307 ms | PauliPropagation.jl 181 ms | **1.7× slower** |
| C_centre_10x10_L4 | 41.5 µs (graph, exact, 3e-17) | PauliPropagation.jl 174 µs (cone, 0e+00) | **4.2×** | 19.6 ms | Lightning 9.5 ms | **2.06× slower** |
| C_centre_10x10_L6 | refused | quimb 216 ms (cone, no ref) | — | — | quimb 216 ms | — |
| C_centre_10x10_L8 | refused | quimb 1.84 s (split, no ref) | — | — | quimb 1.84 s | — |
| C_centre_32x32_L6 | refused | quimb 211 ms (split, no ref) | — | — | quimb 216 ms | — |
| D_noisy1q_2x5_L4 | 25.9 µs (graph, exact, 1e-16) | PauliPropagation.jl 366 µs (whole, 4e-16) | 14.1× | 3.4 ms | PauliPropagation.jl 37.7 ms | 11.1× |
| D_noisy1q_6x6_L3 | 28.5 µs (graph, exact, 3e-16) | PauliPropagation.jl 455 µs (cone, 1e-16) | 16× | 9.72 ms | PauliPropagation.jl 38.2 ms | **3.93×** |
| D_noisy1q_6x6_L5 | 636 µs (graph, exact, 0e+00) | PauliPropagation.jl 14 s (whole, 1e-11) | 2.2e+04× | 41.9 ms | PauliPropagation.jl 14 s | 335× |
| D_noisy1q_20x20_L3 | 149 µs (graph, exact, 2e-16) | PauliPropagation.jl 31.2 ms (split, 4e-16) | 209× | 138 ms | PauliPropagation.jl 122 ms | **1.13× slower** |
| D_noisy1q_32x32_L3 | 324 µs (graph, exact, 0e+00) | PauliPropagation.jl 38 ms (split, 4e-16) | 117× | 360 ms | PauliPropagation.jl 203 ms | **1.77× slower** |
| D_noisy1q_6x6_L6 | refused | none finished | — | — | — | — |
| D_noisy1q_10x10_L6 | refused | none finished | — | — | — | — |
| K_eagle_mz_S5_t0pi8 | 285 µs (graph, 1e-12, 0e+00) | PauliPropagation.jl 11.3 ms (cone, 0e+00) | 39.5× | 64.5 ms | PauliPropagation.jl 52.8 ms | **1.22× slower** |
| K_eagle_mz_S5_t1pi8 | 81.2 ms (native, exact, 0e+00) | none finished | — | 139 ms | — | — |
| K_eagle_mz_S5_t2pi8 | 84.7 ms (native, 1e-12, 0e+00) | none finished | — | 144 ms | — | — |
| K_eagle_mz_S5_t3pi8 | 86 ms (native, 1e-12, 0e+00) | none finished | — | 147 ms | — | — |
| K_eagle_mz_S5_t4pi8 | 8.67 ms (graph, 1e-12, 0e+00) | PauliPropagation.jl 142 ms (split, 0e+00) | 16.4× | 72.7 ms | PauliPropagation.jl 231 ms | **3.17×** |
| K_eagle_z62_S20_t1pi8 | refused | none finished | — | — | — | — |

## Gradients of every shared parameter, repeated

No other tool returns gradients through noise channels; those rows show QCS alone.

| Workload | QCS (backend, mode) | Fastest other (setup) | QCS lead |
|---|---|---|---|
| A_vqe_3x4_L2 | 36.2 µs (graph, exact) | PauliPropagation.jl 588 µs (cone) | 16.2× |
| A_vqe_4x5_L2 | 38.1 µs (graph, exact) | PauliPropagation.jl 1.13 ms (whole) | 29.6× |
| A_vqe_4x5_L4 | 2.09 ms (graph, exact) | Lightning 3.53 s (whole) | 1.69e+03× |
| B_noisy_2x5_L4 | 50.4 µs (graph, exact) | none | — |
| B_noisy_6x6_L3 | 47.9 µs (graph, exact) | none | — |
| B_noisy_6x6_L5 | 1.44 ms (graph, exact) | none | — |
| B_noisy_20x20_L3 | 208 µs (graph, exact) | none | — |
| B_noisy_32x32_L3 | 466 µs (graph, exact) | none | — |
| C_centre_10x10_L4 | 64 µs (graph, exact) | PauliPropagation.jl 1.54 ms (split) | 24.1× |
| D_noisy1q_2x5_L4 | 53.1 µs (graph, exact) | none | — |
| D_noisy1q_6x6_L3 | 49.2 µs (graph, exact) | none | — |
| D_noisy1q_6x6_L5 | 1.74 ms (graph, exact) | none | — |
| D_noisy1q_20x20_L3 | 216 µs (graph, exact) | none | — |
| D_noisy1q_32x32_L3 | 458 µs (graph, exact) | none | — |
| K_eagle_mz_S5_t0pi8 | 686 µs (graph, 1e-12) | PauliPropagation.jl 82.5 ms (whole) | 120× |
| K_eagle_mz_S5_t1pi8 | 92.2 ms (native, exact) | none | — |
| K_eagle_mz_S5_t2pi8 | 92.4 ms (native, exact) | none | — |
| K_eagle_mz_S5_t3pi8 | 97.6 ms (native, 1e-12) | none | — |
| K_eagle_mz_S5_t4pi8 | 21.8 ms (graph, 1e-12) | PauliPropagation.jl 394 ms (split) | 18.1× |

## Every configuration, repeated evaluation

| Workload | QCS graph 1e-12 | QCS graph exact | QCS native 1e-12 | QCS native exact | Aer cone | Aer split | Aer whole | Lightning cone | Lightning split | Lightning whole | PauliPropagation.jl cone | PauliPropagation.jl split | PauliPropagation.jl whole | Qulacs cone | Qulacs split | Qulacs whole | pauli-prop cone | pauli-prop split | pauli-prop whole | quimb cone | quimb split | quimb whole | spd cone | spd split | spd whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_vqe_3x4_L2 | 37.5 µs | 19.4 µs | 204 µs | 198 µs | 7.99 ms | 34.8 ms | 4.59 ms | 6.19 ms | 60 ms | 6.45 ms | 133 µs | 536 µs | 116 µs | 1.52 ms | 577 µs | 1.53 ms | 3.82 ms | 30.7 ms | 3.84 ms | 292 ms | 366 ms | 295 ms | 5.94 ms | 28.9 ms | 5.82 ms |
| A_vqe_4x5_L2 | 52.9 µs | 20.9 µs | 206 µs | 194 µs | 419 ms | 46.8 ms | 422 ms | 349 ms | 115 ms | 351 ms | 221 µs | 1.01 ms | 240 µs | 679 ms | 1.18 ms | 686 ms | 7.09 ms | 56.2 ms | 7.15 ms | 595 ms | 706 ms | 583 ms | 10.2 ms | 52.6 ms | 10.1 ms |
| A_vqe_4x5_L4 | 472 µs | 709 µs | 27.2 ms | 25.5 ms | 553 ms | 639 ms | 551 ms | 422 ms | 584 ms | 426 ms | 2.16 s | timeout | 2.52 s | 1.1 s | 704 ms | 1.1 s | timeout | timeout | timeout | 6.84 s | 7.15 s | 6.76 s | timeout | timeout | timeout |
| A_vqe_4x6_L6 | — | refused | timeout | refused | 14 s | timeout | 14.1 s | timeout | timeout | timeout | failed | failed | failed | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| B_noisy_2x5_L4 | 46.5 µs | 27.8 µs | 294 µs | 283 µs | 846 ms | 110 ms | 850 ms | — | — | — | 381 µs | 1.16 ms | 379 µs | 5.43 s | 127 ms | 5.48 s | 15.4 ms | 48.8 ms | 15.9 ms | — | — | — | — | — | — |
| B_noisy_6x6_L3 | 46.8 µs | 27.1 µs | 344 µs | 337 µs | — | 486 ms | — | — | — | — | 461 µs | 1.12 ms | 671 µs | — | 709 ms | — | 12.2 ms | 47.7 ms | 12.2 ms | — | — | — | — | — | — |
| B_noisy_6x6_L5 | 657 µs | 631 µs | 24.6 ms | 25.6 ms | — | — | — | — | — | — | 13.9 s | timeout | 15.9 s | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| B_noisy_20x20_L3 | — | 146 µs | 4.32 ms | 4.46 ms | — | 7.04 s | — | — | — | — | 74.8 ms | 14.8 ms | 78.5 ms | — | 10.8 s | — | 1.02 s | 558 ms | 1.04 s | — | — | — | — | — | — |
| B_noisy_32x32_L3 | — | 321 µs | 10.1 ms | 9.67 ms | — | timeout | — | — | — | — | 1.14 s | 39.1 ms | 1.44 s | — | timeout | — | timeout | 1.43 s | timeout | — | — | — | — | — | — |
| C_centre_10x10_L4 | 61.3 µs | 41.5 µs | 257 µs | 252 µs | 11 ms | 11.1 ms | — | 8.66 ms | 8.74 ms | — | 174 µs | 177 µs | 1.16 ms | 10.2 ms | 10.2 ms | — | 10.1 ms | 10.1 ms | 42.5 ms | 53.4 ms | 55.3 ms | 284 ms | 6.17 ms | 6.07 ms | 45.7 ms |
| C_centre_10x10_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | 216 ms | 301 ms | 644 ms | timeout | timeout | timeout |
| C_centre_10x10_L8 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | 2.19 s | 1.84 s | 4.88 s | timeout | timeout | timeout |
| C_centre_32x32_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | timeout | — | — | — | timeout | timeout | timeout | 400 ms | 211 ms | 4.83 s | timeout | timeout | timeout |
| D_noisy1q_2x5_L4 | 44.5 µs | 25.9 µs | 306 µs | 298 µs | 779 ms | 97.2 ms | 777 ms | — | — | — | 390 µs | 1.13 ms | 366 µs | 2.01 s | 127 ms | 2.02 s | 15 ms | 49.3 ms | 15.1 ms | — | — | — | — | — | — |
| D_noisy1q_6x6_L3 | 49.5 µs | 28.5 µs | 395 µs | 396 µs | — | 491 ms | — | — | — | — | 455 µs | 1.07 ms | 461 µs | — | 711 ms | — | 11 ms | 49.2 ms | 11.1 ms | — | — | — | — | — | — |
| D_noisy1q_6x6_L5 | 661 µs | 636 µs | 26.1 ms | 25.5 ms | — | — | — | — | — | — | 14.3 s | timeout | 14 s | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| D_noisy1q_20x20_L3 | — | 149 µs | 5.27 ms | 5.34 ms | — | 7.03 s | — | — | — | — | 80.5 ms | 31.2 ms | 76.9 ms | — | 10.8 s | — | 918 ms | 566 ms | 918 ms | — | — | — | — | — | — |
| D_noisy1q_32x32_L3 | — | 324 µs | 11.5 ms | 11.8 ms | — | timeout | — | — | — | — | 1.17 s | 38 ms | 1.21 s | — | timeout | — | timeout | 1.43 s | timeout | — | — | — | — | — | — |
| D_noisy1q_6x6_L6 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| D_noisy1q_10x10_L6 | — | refused | timeout | refused | — | — | — | — | — | — | timeout | failed | timeout | — | — | — | timeout | timeout | timeout | — | — | — | — | — | — |
| K_eagle_mz_S5_t0pi8 | 285 µs | — | 5.55 ms | 8.1 ms | — | — | — | — | — | — | 11.3 ms | 60.1 ms | 11.6 ms | — | — | — | 92.4 ms | 339 ms | 92.3 ms | timeout | timeout | timeout | 139 ms | timeout | 140 ms |
| K_eagle_mz_S5_t1pi8 | — | — | 81.7 ms | 81.2 ms | — | — | — | — | — | — | timeout | timeout | timeout | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t2pi8 | — | — | 84.7 ms | 85.4 ms | — | — | — | — | — | — | failed | timeout | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t3pi8 | — | — | 86 ms | 86 ms | — | — | — | — | — | — | failed | timeout | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
| K_eagle_mz_S5_t4pi8 | 8.67 ms | — | 13.1 ms | 26.5 ms | — | — | — | — | — | — | 201 ms | 142 ms | 190 ms | — | — | — | 3.38 s | 2.68 s | 3.44 s | timeout | timeout | timeout | 2.44 s | timeout | 2.43 s |
| K_eagle_z62_S20_t1pi8 | refused | refused | refused | refused | — | — | — | — | — | — | failed | failed | failed | — | — | — | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout | timeout |
