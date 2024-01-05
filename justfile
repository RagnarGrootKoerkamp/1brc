run:
    cargo run -r
small:
    cargo run -r -- measurements-small.txt
flame:
    cargo flamegraph --open
stat:
    cargo build -r --quiet
    perf stat cargo run -r
report:
    perf report
