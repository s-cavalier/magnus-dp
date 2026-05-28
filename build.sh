python gen_nodes.py 64
git submodule update --init --recursive
cmake -S . -B build
(
    cd build
    make
)
(
    cd test
    python -c "import magnus; magnus.max_order(); print('Smoke test succeeded. Package should be good to go.')"
)