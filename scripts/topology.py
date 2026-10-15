# SPDX-License-Identifier: Apache-2.0
"""TP size, GLM-5.3 sharding math and per-rank memory projections (CPU only)."""
from fp4_kv_layout import CARVE_BYTES, layout

SUPPORTED_TP = (4, 8)
DEFAULT_TP = 4
GiB = 1 << 30

# Pinned target config (tests/fixtures/boot_preflight_glm53_config.json).
MODEL = dict(hidden=6144, heads=64, q_lora=2048, kv_lora=512, rope=64, nope=192, v_head=256,
             dense_intermediate=12288, moe_intermediate=2048, routed_experts=256, shared_experts=1,
             vocab=154880, layers=78, mtp_layers=1, index_heads=32, index_head_dim=128,
             index_layers=22, max_position=1048576)
MARLIN_N, MARLIN_K, EXPERT_GROUP = 64, 128, 128
# Split MLA kernels tile heads by 16 and mask the rest.
MLA_HEADS_PER_RANK = (8, 16)

# Loader report, every rank, TP4 release (docs/headroom/PLAN.md).
TP4_WEIGHT_GIB = 96.83
SLOTS = 4
KSTOP_LOOKAHEAD = 3
BLOCK_TOKENS = 64
RELEASE_CONTEXT = 262144
TP4_HEAD_BYTES = 6318718976


def tp_size(env):
    text = (env.get('RECIPE_TP_SIZE', '') or str(DEFAULT_TP)).strip()
    if not text.isdigit() or int(text) not in SUPPORTED_TP:
        raise ValueError('RECIPE_TP_SIZE must be one of ' + ', '.join(map(str, SUPPORTED_TP)))
    return int(text)


def shards(tp):
    """Per-rank partition of every TP-sharded GEMM, with its kernel constraint."""
    m = MODEL
    heads = m['heads'] // tp if m['heads'] % tp == 0 else None
    rows = []

    def add(name, total, sharded_dim, other, rule):
        per = total // tp if total % tp == 0 else None
        if per is None:
            ok = False
        elif rule == 'n':
            ok = per % MARLIN_N == 0 and other % MARLIN_K == 0
        elif rule == 'k':
            ok = per % MARLIN_K == 0 and other % MARLIN_N == 0
        elif rule == 'k-group':
            ok = per % MARLIN_K == 0 and per % EXPERT_GROUP == 0 and other % MARLIN_N == 0
        else:
            ok = True
        rows.append(dict(name=name, total=total, per_rank=per, dim=sharded_dim, ok=ok, rule=rule))

    add('attention heads', m['heads'], 'heads', 0, 'split')
    add('q_b_proj N', m['heads'] * (m['nope'] + m['rope']), 'N', m['q_lora'], 'n')
    add('kv_b_proj N', m['heads'] * (m['nope'] + m['v_head']), 'N', m['kv_lora'], 'n')
    add('o_proj K', m['heads'] * m['v_head'], 'K', m['hidden'], 'k')
    add('dense gate_up N', 2 * m['dense_intermediate'], 'N', m['hidden'], 'n')
    add('dense down K', m['dense_intermediate'], 'K', m['hidden'], 'k')
    add('expert gate_up N', 2 * m['moe_intermediate'], 'N', m['hidden'], 'n')
    add('expert down K', m['moe_intermediate'], 'K', m['hidden'], 'k-group')
    add('shared expert down K', m['shared_experts'] * m['moe_intermediate'], 'K', m['hidden'], 'k')
    add('LM head N (draft NVFP4 bank)', m['vocab'], 'N', m['hidden'], 'n')
    add('vocab embedding rows', m['vocab'], 'rows', 0, 'split')
    mla = heads in MLA_HEADS_PER_RANK
    rows.append(dict(name='Triton MLA heads per rank', total=m['heads'], per_rank=heads, dim='heads',
                     ok=mla, rule='kernel tile 16, admitted ' + '/'.join(map(str, MLA_HEADS_PER_RANK))))
    return rows


def replicated_bytes():
    """Estimated per-rank bytes of replicated weights (identical at every TP size)."""
    m = MODEL
    qkv_a = (m['layers'] + m['mtp_layers']) * m['hidden'] * (m['q_lora'] + m['kv_lora'] + m['rope'])
    index_params = (m['q_lora'] * m['index_heads'] * m['index_head_dim'] + m['hidden'] * m['index_head_dim']
                    + m['hidden'] * m['index_heads'])
    indexer = m['index_layers'] * index_params * 2
    moe_layers = m['layers'] - 3 + m['mtp_layers']
    router = moe_layers * m['hidden'] * m['routed_experts'] * 2
    eh_proj = m['hidden'] * 2 * m['hidden']
    return dict(fused_qkv_a=qkv_a, indexer=indexer, router=router, eh_proj=eh_proj,
                total=qkv_a + indexer + router + eh_proj)


def weights_per_rank_gib(tp, replicated_gib=None):
    rep = replicated_bytes()['total'] / GiB if replicated_gib is None else replicated_gib
    return rep + (TP4_WEIGHT_GIB - rep) * 4 / tp


def slot_blocks(context=RELEASE_CONTEXT):
    return -(-(context + KSTOP_LOOKAHEAD) // BLOCK_TOKENS)


def head_for_blocks(blocks, mode='fp4x'):
    """Smallest 2 MiB-aligned ordinary head whose FP4x pool holds ``blocks``."""
    unit = 2 << 20
    probe = layout(mode)
    head = max(GiB, -(-(blocks * probe['block_bytes'] - CARVE_BYTES) // unit) * unit)
    while layout(mode, head)['blocks'] < blocks:
        head += unit
    return head


def full_slot_head(slots=SLOTS, context=RELEASE_CONTEXT):
    """Ordinary head that lets every slot hold one maximum-context request."""
    return head_for_blocks(slots * slot_blocks(context) + 6)


def kv_plan(head_bytes, context=RELEASE_CONTEXT):
    pool = layout('fp4x', head_bytes)
    return dict(head_bytes=head_bytes, blocks=pool['blocks'], tokens=pool['tokens'],
                pool_gib=pool['aligned_bytes'] / GiB,
                full_context_slots=(pool['blocks'] - 1) // slot_blocks(context),
                max_single_context=min(int(pool['max_model_len']), MODEL['max_position']))


def summary(tp, head_bytes):
    weights = weights_per_rank_gib(tp)
    plan = kv_plan(head_bytes)
    return dict(tp=tp, weights_gib=round(weights, 2), freed_vs_tp4_gib=round(weights_per_rank_gib(4) - weights, 2),
                replicated_gib=round(replicated_bytes()['total'] / GiB, 2),
                kv=plan, kv_growth_vs_tp4_gib=round((head_bytes - TP4_HEAD_BYTES) / GiB, 2),
                shards_ok=all(r['ok'] for r in shards(tp)))


if __name__ == '__main__':
    import json
    import sys
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TP
    head = TP4_HEAD_BYTES if tp == 4 else full_slot_head()
    print(json.dumps(dict(summary(tp, head), shards=shards(tp)), indent=2))
