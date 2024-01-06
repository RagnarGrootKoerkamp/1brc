run:
    cargo run -r
small:
    cargo run -- measurements-small.txt
flame:
    cargo flamegraph --open
stat:
    cargo build -r --quiet
    perf stat -d cargo run -r
record:
    cargo build -r --quiet
    perf record cargo run -r
report:
    perf report
