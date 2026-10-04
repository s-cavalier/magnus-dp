git submodule update --init --recursive
MAGNUS_FIXED_DIMS="${MAGNUS_FIXED_DIMS:-2;3;4}"
MAGNUS_BUILD_JOBS="${MAGNUS_BUILD_JOBS:-1}"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    "-DMAGNUS_FIXED_DIMS=${MAGNUS_FIXED_DIMS}" \
    -G Ninja
(
    cd build
)
ninja -C build -j "${MAGNUS_BUILD_JOBS}"
(
    PYTHONPATH=. python -c "import magnus; magnus.max_order(); print('Smoke test succeeded. Package should be good to go.')"
)
