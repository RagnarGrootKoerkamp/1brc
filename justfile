run:
    cargo run -r
small:
    cargo run -r -- measurements-small.txt
flame:
    cargo flamegraph --open
