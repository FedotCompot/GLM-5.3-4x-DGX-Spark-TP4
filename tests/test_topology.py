"""TP size selection, sharding divisibility and per-rank memory projections."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import topology as T
from fp4_kv_layout import layout


class Topology(unittest.TestCase):
    def test_model_constants_match_pinned_config(self):
        c = json.loads((ROOT / 'tests/fixtures/boot_preflight_glm53_config.json').read_text())
        m = T.MODEL
        pairs = dict(hidden='hidden_size', heads='num_attention_heads', q_lora='q_lora_rank',
                     kv_lora='kv_lora_rank', rope='qk_rope_head_dim', nope='qk_nope_head_dim',
                     v_head='v_head_dim', dense_intermediate='intermediate_size',
                     moe_intermediate='moe_intermediate_size', routed_experts='n_routed_experts',
                     shared_experts='n_shared_experts', vocab='vocab_size', layers='num_hidden_layers',
                     mtp_layers='num_nextn_predict_layers', index_heads='index_n_heads',
                     index_head_dim='index_head_dim', max_position='max_position_embeddings')
        for key, name in pairs.items():
            self.assertEqual(m[key], c[name], key)
        self.assertEqual(c['indexer_types'].count('full') + m['mtp_layers'], m['index_layers'])

    def test_tp_size_selection(self):
        self.assertEqual(T.tp_size({}), 4)
        self.assertEqual(T.tp_size({'RECIPE_TP_SIZE': ''}), 4)
        self.assertEqual(T.tp_size({'RECIPE_TP_SIZE': '8'}), 8)
        for bad in ('2', '16', '6', 'eight', '-8'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                T.tp_size({'RECIPE_TP_SIZE': bad})

    def test_tp4_every_partition_is_legal(self):
        rows = {r['name']: r for r in T.shards(4)}
        self.assertTrue(all(r['ok'] for r in rows.values()))
        self.assertEqual(rows['attention heads']['per_rank'], 16)
        self.assertEqual(rows['expert down K']['per_rank'], 512)
        self.assertEqual(rows['LM head N (draft NVFP4 bank)']['per_rank'], 38720)

    def test_tp8_only_the_draft_head_bank_breaks(self):
        rows = {r['name']: r for r in T.shards(8)}
        broken = [name for name, r in rows.items() if not r['ok']]
        self.assertEqual(broken, ['LM head N (draft NVFP4 bank)'])
        self.assertEqual(rows['LM head N (draft NVFP4 bank)']['per_rank'] % 64, 32)
        self.assertEqual(rows['attention heads']['per_rank'], 8)
        self.assertEqual(rows['expert gate_up N']['per_rank'], 512)
        self.assertEqual(rows['expert down K']['per_rank'], 256)
        self.assertEqual(rows['kv_b_proj N']['per_rank'], 3584)
        self.assertTrue(rows['Triton MLA heads per rank']['ok'])

    def test_unsupported_tp_shapes_fail(self):
        rows = {r['name']: r for r in T.shards(16)}
        self.assertFalse(rows['Triton MLA heads per rank']['ok'])
        self.assertTrue(rows['expert down K']['ok'])
        self.assertFalse({r['name']: r for r in T.shards(32)}['expert down K']['ok'])
        self.assertFalse({r['name']: r for r in T.shards(3)}['attention heads']['ok'])

    def test_weight_projection(self):
        self.assertAlmostEqual(T.weights_per_rank_gib(4), T.TP4_WEIGHT_GIB)
        rep = T.replicated_bytes()['total'] / T.GiB
        self.assertTrue(1.0 < rep < 3.0, rep)
        tp8 = T.weights_per_rank_gib(8)
        self.assertAlmostEqual(tp8, rep + (T.TP4_WEIGHT_GIB - rep) / 2)
        for guess in (1.0, 3.0):
            self.assertLess(abs(T.weights_per_rank_gib(8, guess) - tp8), 1.0)

    def test_tp4_release_pool_is_reproduced(self):
        plan = T.kv_plan(T.TP4_HEAD_BYTES)
        self.assertEqual((plan['blocks'], plan['tokens']), (4135, 264640))
        self.assertEqual(plan['full_context_slots'], 1)

    def test_tp8_head_holds_four_max_contexts(self):
        head = T.full_slot_head()
        self.assertEqual(head, 31404851200)
        self.assertEqual(head % (2 << 20), 0)
        plan = T.kv_plan(head)
        self.assertEqual(plan['full_context_slots'], 4)
        self.assertEqual(plan['blocks'], 4 * T.slot_blocks() + 6)
        self.assertLess(layout('fp4x', head - (2 << 20))['blocks'], plan['blocks'])
        self.assertEqual(plan['max_single_context'], T.MODEL['max_position'])

    def test_tp8_kv_growth_fits_inside_freed_weights(self):
        s = T.summary(8, T.full_slot_head())
        self.assertGreater(s['freed_vs_tp4_gib'] - s['kv_growth_vs_tp4_gib'], 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
