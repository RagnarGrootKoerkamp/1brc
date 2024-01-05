run:
    cargo run -r
small:
    cargo run -r -- measurements-small.txt
flame:
    cargo flamegraph --open
stat:
    cargo build -r --quiet
    perf stat -d cargo run -r
report:
    perf report
