import argparse
import struct
import numpy as np
from numpy.polynomial.legendre import leggauss

MAGIC = b"LG01"

HEADER_FMT = "<4sIQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def build_bytes(max_n: int):
    offsets = [0]
    all_nodes = []
    all_weights = []

    cursor = 0

    for n in range(1, max_n + 1):
        offsets.append(cursor)

        nodes, weights = leggauss(n)

        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights

        all_nodes.extend(nodes)
        all_weights.extend(weights)

        cursor += n

    offsets_offset = HEADER_SIZE
    nodes_offset = offsets_offset + 8 * (max_n + 1)
    weights_offset = nodes_offset + 8 * len(all_nodes)

    return b"".join([
        struct.pack(
            HEADER_FMT,
            MAGIC,
            max_n,
            offsets_offset,
            nodes_offset,
            weights_offset,
        ),
        struct.pack(f"<{len(offsets)}Q", *offsets),
        struct.pack(f"<{len(all_nodes)}d", *all_nodes),
        struct.pack(f"<{len(all_weights)}d", *all_weights),
    ])


def build_file(max_n: int, out_path: str):
    data = build_bytes(max_n)

    with open(out_path, "w") as f:
        f.write("#include <cstddef>\n\n")
        f.write("extern \"C\" {\n")
        f.write("    alignas(8) std::byte _binary_gl_nodes_bin_start[] = {\n")

        for i in range(0, len(data), 12):
            chunk = data[i:i + 12]
            values = ", ".join(f"std::byte{{0x{b:02x}}}" for b in chunk)
            f.write(f"        {values},\n")

        f.write("    };\n")
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("out_path", nargs="?", default="gl-nodes.cpp")

    args = parser.parse_args()

    if args.N < 1: raise ValueError("N must be >= 1")

    build_file(args.N, args.out_path)


if __name__ == "__main__":
    main()
