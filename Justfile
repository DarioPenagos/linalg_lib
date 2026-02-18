# Builds project with cargo build
build:
    cargo build

# Builds the project, then runs the tests with the latest .so file
test: build
    cp target/debug/liblinalg_lib.so tests/linalg_lib.so
    pytest

check:
    cargo check

repl: build
    cp target/debug/linalg_lib.so .
    python -i -c "from linalg_lib import *"
    rm linalg_lib.so
