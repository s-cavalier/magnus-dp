#!/usr/bin/env bash

set -u
set -o pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$root_dir/magnus"
tests_dir="$root_dir/tests"
build_marker="$(find "$package_dir" -maxdepth 1 -type f -name '_core*.so' 2>/dev/null | head -n 1)"

if [[ -z "${build_marker}" ]]; then
    bash "$root_dir/build.sh"
fi

status=0
for test_file in "$tests_dir"/*.py; do
    echo "Running ${test_file#$root_dir/}"
    if [[ "$(basename "$test_file")" == test_*.py ]]; then
        PYTHONPATH="$root_dir" python -m pytest "$test_file" "$@" || status=1
    else
        PYTHONPATH="$root_dir" python "$test_file" || status=1
    fi
done

exit "$status"
