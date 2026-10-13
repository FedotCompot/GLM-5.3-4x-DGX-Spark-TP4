#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Estimated phase savings, including stock metadata splits and FP4x Q tiles.

No fleet measurements. At 95K both GPU busy and indexer time are interpolated.
35K indexer = 10.8 s * 8.5%; 129K = 2.15+0.98 s, not rounded 25%.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/bringup'))
from glm_fp4_prefill import geometry
from glm_indexer_row_shard import tile_plan


def estimate(depth, rows, layers, gather_ms):
    t=(depth-35000)/(129000-35000)
    index=(.918+(3.13-.918)*t)*rows/8500
    busy=(10.8+(13.4-10.8)*t)*rows/8500
    cap=(512<<20)//(4*depth)  # pinned metadata builder's default logits budget
    spans=[min(cap,rows-offset) for offset in range(0,rows,cap)]
    qt,_=geometry(depth)
    tasks,owners,slices=tile_plan(spans,[qt]*len(spans))
    # All tiles in a single request retain exactly the same scorer shape.
    # Final partially valid tiles still run full padded logits. Top-k varies
    # by valid rows, so this equal-tile approximation is deliberately explicit.
    removed=1-max(b-a for a,b in owners)/len(tasks)
    saved=index*removed
    gather=layers*gather_ms/1000
    return dict(depth=depth,scheduled_rows=rows,layer_budget=layers,gather_ms_per_layer=gather_ms,
                metadata_rows=spans,stock_q_tile=qt,stock_tiles=len(tasks),
                max_owner_tiles=max(b-a for a,b in owners),estimated_removed_fraction=removed,
                index_s=index,busy_s=busy,saved_index_s=saved,gather_s=gather,
                net_saved_s=saved-gather,net_busy_reduction_pct=100*(saved-gather)/busy,
                max_shard_bytes=max(b-a for a,b in slices)*2048*4,
                break_even_gather_ms=saved*1000/layers)


if __name__=='__main__':
    print(json.dumps(dict(scope='ESTIMATE_ONLY; checked probe has no speedup',
        assumptions=dict(profile_new_tokens=8500,metadata_logits_budget_mib=512,
            gather_ms_range=[.5,3.0],ideal_tp4_removed_fraction=.75,
            layer_budgets='78 pessimistic; 21 expected computing target layers, confirm live'),
        estimates=[estimate(d,r,l,g) for d in (35000,95000,129000)
                   for r in (2048,4096) for l in (78,21) for g in (.5,3.0)]),indent=2))
