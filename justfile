run *args:
    cargo run -r --quiet -- {{args}}
large *args:
    cargo run -r --quiet -- measurements3.txt {{args}}
time *args:
    cargo build -r --quiet && time ./target/release/one-billion-row-challenge {{args}}
small:
    cargo run --quiet -- measurements-small.txt
flame:
    cargo flamegraph --open
stat *args:
    cargo build -r --quiet
    perf stat cargo run -r --quiet -- {{args}}
record:
    cargo build -r --quiet
    perf record cargo run -r
report:
    perf report

test *args:
    cargo test --quiet -- {{args}}

verify:
    cargo run -r --quiet -- --print > result.txt && diff result.txt result_ref.txt

wverify:
    cargo run -r --quiet -- --print > result.txt && wdiff result.txt result_ref.txt

cpu-slow:
    sudo cpupower frequency-set --governor performance -d 0.6GHz -u 3.6GHz > /dev/null
cpu-fast:
    sudo cpupower frequency-set --governor performance -d 0.6GHz -u 4.6GHz > /dev/null
