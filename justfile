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

test *args:
    cargo test --quiet -- {{args}}

verify:
    cargo run -r --quiet -- --print > results.txt && diff result.txt result_ref.txt
