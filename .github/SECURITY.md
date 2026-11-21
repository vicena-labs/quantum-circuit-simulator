# Security and resource boundaries

The library is for trusted local research workloads. It is not a hardened multi-tenant service, formal verifier or operating-system sandbox. Large circuits can consume substantial CPU/RAM even when a per-observable expansion cap is set.

The CLI limits input size and checks duplicate/unknown JSON fields, numeric finiteness, wire/parameter ranges and operation support. Its parent process kills an over-deadline worker; input reading itself is outside that deadline. Resource caps do not bound aggregate process memory. A service operator must additionally enforce OS memory, CPU, file-size, request-rate and identity boundaries.

Use installed native binaries from a trusted build. There is no compiler execution at runtime or pickle deserialization. Arbitrary user-defined matrices and importable code are not accepted by the JSON interface.

The experimental C++ runners have a narrower custom parser and plan format and are not hardened for hostile requests. Keep them separate from network services.

Report suspected vulnerabilities privately through GitHub's private vulnerability reporting on [the repository](https://github.com/vicena-labs/quantum-circuit-simulator/security/advisories/new), not in a public issue. No response-time promise is made.
