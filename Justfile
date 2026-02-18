# Builds project with cargo build
build:
    cargo build
    cp target/debug/liblinalg_lib.so target/debug/linalg_lib.so

# Builds the project, then runs the tests with the latest .so file
test: build
    cp target/debug/liblinalg_lib.so tests/linalg_lib.so
    pytest

repl: build
    cp target/debug/linalg_lib.so .
    python -i -c "from linalg_lib import *"
    rm linalg_lib.so
