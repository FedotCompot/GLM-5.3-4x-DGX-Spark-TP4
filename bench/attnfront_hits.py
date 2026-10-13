#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline validator of coordinator-collected per-worker B-boot logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PIECES = ('parts', 'norms', 'rope_pack')


def check(text, layers=78):
    found = set()
    for piece, rows, name in re.findall(r'ATTNFRONT_HIT (\w+) M(\d+) (\S+)', text):
        layer = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
        if layer:
            found.add((piece, int(rows), int(layer[1])))
    expected = {(piece, m, layer) for piece in PIECES
                for m in range(1, 17) for layer in range(1, layers)}
    missing = sorted(expected - found)
    zero = sorted(cell for cell in found if cell[2] == 0)
    if 'ATTNFRONT_READY' not in text or missing or zero:
        raise ValueError(f'HOLD: missing={len(missing)} layer0_hits={len(zero)} '
                         f'first_missing={missing[:8]}')
    return dict(required_hits=len(expected), observed_target_hits=len(found & expected),
                mtp_hits=len([cell for cell in found if cell[2] >= layers]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='append', required=True, help='r0=LOG (repeat r0..r3)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    workers = dict(item.split('=', 1) for item in args.worker)
    if len(args.worker) != 4 or set(workers) != {'r0', 'r1', 'r2', 'r3'}:
        raise ValueError('require exactly four distinct workers r0..r3')
    if len({Path(p).resolve() for p in workers.values()}) != 4:
        raise ValueError('require separate per-worker logs')
    records = {}
    for rank, path in workers.items():
        data = Path(path).read_bytes()
        records[rank] = dict(check(data.decode()), log_sha256=hashlib.sha256(data).hexdigest())
    args.out.write_text(json.dumps(dict(status='PASS', workers=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
