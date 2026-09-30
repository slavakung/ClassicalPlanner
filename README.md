# Serial planner source and bounded benchmarks

This source-only distribution contains all 26 serial planning algorithms, all
C++ and Python tests, and **1041 selected algorithm/problem pairs**
from 1248 considered pairs. Selected runs completed without an error
under a **2 GiB virtual-address-space limit**, with measured peak memory below
that limit and a **120-second per-case timeout**. Solved
plans passed independent replay. Completion at a configured depth can also mean
bounded-unsolved; that outcome does not establish infeasibility.

Fresh qualification attempted 1097 pairs; 151 pairs were excluded using earlier timeout evidence and were not rerun.

The selection is [benchmarks/public_cases.json](benchmarks/public_cases.json).
Each selected pair records its completion status, measured memory and applied
limits in that manifest. Checks may use a stricter 512 MiB limit; completion
under that cap also qualifies below 2 GiB. All 48 generated input fixtures remain
available for regression and generator tests; only the explicit selected pairs
constitute this benchmark suite. Source
files, test support and fixture definitions are included. Compiled binaries,
historical results and live campaign files are excluded.

A C++23 compiler, CMake 3.24 or newer and Python 3.9 or newer are required.
Build and run the tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --parallel 2
python3 -m unittest discover -s tests -p 'test_*.py'
python3 benchmarks/generate_instances.py --check
```

Run the selected benchmarks on Linux with the recorded per-case limits:

```sh
python3 benchmarks/run_bounded.py --cases benchmarks/public_cases.json --output benchmark-results
```

The runner defaults to `build/benchmark_planner` and runs explicit pairs only.
It reapplies the memory and time limits; performance and memory can vary with
the compiler, operating system and machine. Qualification describes the measured
configuration and is not a guarantee for every build. It creates fresh local
records in `benchmark-results`. Optional plotting support uses
`benchmarks/requirements.txt`; the build and bounded runner need no Python
packages. Legacy campaign/report modules remain as regression-test dependencies.

[BUILD_PUBLIC_MANIFEST.json](BUILD_PUBLIC_MANIFEST.json) records the qualification
evidence digest and SHA-256 inventory of every included file except itself.
Licensed under [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
