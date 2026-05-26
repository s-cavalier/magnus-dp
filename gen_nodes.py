import argparse
import struct
import numpy as np
from numpy.polynomial.legendre import leggauss

MAGIC = b"LG01"

HEADER_FMT = "<4sIQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def build_file(max_n: int, out_path: str):
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

    with open(out_path, "wb") as f:
        f.write(struct.pack(
            HEADER_FMT,
            MAGIC,
            max_n,
            offsets_offset,
            nodes_offset,
            weights_offset,
        ))

        f.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        f.write(struct.pack(f"<{len(all_nodes)}d", *all_nodes))
        f.write(struct.pack(f"<{len(all_weights)}d", *all_weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)

    args = parser.parse_args()

    if args.N < 1:
        raise ValueError("N must be >= 1")

    build_file(args.N, 'gl-nodes.bin')


if __name__ == "__main__":
    main()