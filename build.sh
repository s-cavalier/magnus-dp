python gen_nodes.py 64
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -G Ninja
(
    cd build
)
ninja -C build
(
    cd demo
    python -c "import magnus; magnus.max_order(); print('Smoke test succeeded. Package should be good to go.')"
)