#!/usr/bin/env bash

set -u
set -o pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$root_dir/magnus"
build_marker="$(find "$package_dir" -maxdepth 1 -type f -name '_core*.so' 2>/dev/null | head -n 1)"

if [[ -z "${build_marker}" ]]; then
    bash "$root_dir/build.sh"
fi

status=0
for test_file in "$package_dir"/tests/*.py; do
    echo "Running ${test_file#$root_dir/}"
    if ! PYTHONPATH="$root_dir" python "$test_file"; then
        echo "FAIL ${test_file#$root_dir/}"
        status=1
    else
        echo "OK ${test_file#$root_dir/}"
    fi
done

exit "$status"
