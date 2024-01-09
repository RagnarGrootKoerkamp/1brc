run *args:
    cargo run -r --quiet -- {{args}}
small:
    cargo run --quiet -- measurements-small.txt
flame:
    cargo flamegraph --open
stat:
    cargo build -r --quiet
    perf stat cargo run -r
record:
    cargo build -r --quiet
    perf record cargo run -r
report:
    perf report
