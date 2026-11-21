# Contributing

Contributions are welcome under the project's Apache-2.0 license. Submit only material you are entitled to contribute and preserve third-party notices.

Work in a virtual environment, install `.[test]`, and run `python -m pytest -q`. A numerical change should include independent small-system density/reference checks and gradient checks, not only comparisons between wrappers around the same kernel. Preserve complete error outputs, refusal cases, source hashes and realistic performance denominators. Optional external comparison contracts are checked with `python -m pytest -q benchmarks/test_comparators.py` after installing `benchmarks/requirements.txt`.

Use synthetic fixtures by default. Do not contribute private calibration data, customer information, credentials or code whose redistribution rights are unclear. Propose breaking API changes before implementation. Request a package build and installed-wheel test, not only an editable-source test.
