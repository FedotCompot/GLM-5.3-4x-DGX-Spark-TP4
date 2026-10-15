"""TP size selection through start.sh, profiles/tp8.env and the launcher's rendered vectors."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import topology

SITE = '''HOSTS=({hosts})
IPS=({ips})
FABRIC_IFACE=eth1
IB_HCA=hca0,hca1
MODEL_DIR=/models/target
NCCL_HOST_DIR=/nccl
OVERLAY_REMOTE=/runtime
NCCL_SHA256={sha}
GLM_ATTN_NVFP4_DIR=/srv/attn
GLM_NVFP4_MORE_DIR=/srv/more
'''


def layered_env(tp, extra='', ranks=None):
    """The environment start.sh hands the launcher, captured by a python3 shim."""
    ranks = tp if ranks is None else ranks
    with tempfile.TemporaryDirectory() as d:
        shim = Path(d, 'python3')
        shim.write_text('#!/usr/bin/env bash\nenv -0 > "$TP8_ENV_OUT"\n')
        shim.chmod(0o755)
        config = Path(d, 'site.env')
        config.write_text(SITE.format(hosts=' '.join('s%d' % (r + 1) for r in range(ranks)),
                                      ips=' '.join('192.0.2.%d' % (r + 1) for r in range(ranks)),
                                      sha='a' * 64) + extra)
        out = Path(d, 'env')
        env = {'PATH': d + ':' + os.environ['PATH'], 'HOME': os.environ.get('HOME', d),
               'RECIPE_CONFIG': str(config), 'TP8_ENV_OUT': str(out)}
        if tp is not None:
            env['RECIPE_TP_SIZE'] = str(tp)
        # Profile LD_PRELOAD names a container path; the shim's loader warns about it.
        run = subprocess.run(['bash', str(ROOT / 'start.sh'), 'serve'], env=env, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        pairs = [item.split('=', 1) for item in out.read_bytes().decode().split('\0') if '=' in item]
    return {k: v for k, v in pairs if k not in ('PATH', 'TP8_ENV_OUT', 'RECIPE_CONFIG')}


def load_cluster():
    spec = importlib.util.spec_from_file_location('cluster_tp', ROOT / 'scripts/cluster.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Selection(unittest.TestCase):
    def test_default_is_tp4_and_profile_untouched(self):
        env = layered_env(None, ranks=4)
        self.assertEqual(env['RECIPE_TP_SIZE'], '4')
        self.assertEqual(env['RECIPE_KV_HEAD_BYTES'], str(topology.TP4_HEAD_BYTES))
        self.assertEqual(env['GLM_DRAFT_HEAD'], 'nvfp4')
        self.assertEqual(env['GLM_MARLIN_L2HINT'], 'on')

    def test_tp8_layers_profile(self):
        env = layered_env(8)
        self.assertEqual(env['RECIPE_HOSTS'].split(), ['s%d' % r for r in range(1, 9)])
        self.assertEqual(env['RECIPE_KV_HEAD_BYTES'], str(topology.full_slot_head()))
        for key, value in (('GLM_DRAFT_HEAD', '0'), ('GLM_DRAFT_HEAD_INIT', '0'), ('GLM_DRAFT_EHPROJ', '0'),
                           ('GLM_DRAFT_EHPROJ_INIT', '0'), ('GLM_MARLIN_L2HINT', 'off'),
                           ('GLM_ATTN_FRONT_FUSE', '0')):
            self.assertEqual(env[key], value, key)
        self.assertEqual(env['RECIPE_MAX_MODEL_LEN'], '262144')

    def test_site_config_still_wins_over_tp8_profile(self):
        env = layered_env(8, 'RECIPE_KV_HEAD_BYTES=8589934592\n')
        self.assertEqual(env['RECIPE_KV_HEAD_BYTES'], '8589934592')

    def test_tp_size_from_site_config(self):
        env = layered_env(None, 'RECIPE_TP_SIZE=8\n', ranks=8)
        self.assertEqual((env['RECIPE_TP_SIZE'], env['GLM_DRAFT_HEAD']), ('8', '0'))


class Rendering(unittest.TestCase):
    def cluster(self, env):
        self.enterContext(patch.dict(os.environ, env, clear=True))
        module = load_cluster()
        fake = types.ModuleType('dispram_recipe')
        fake.container_args = lambda env: (['-v', '/d/run:/run/dispram'], {'GLM_DISPRAM_KV': 'require'})
        self.enterContext(patch.dict(sys.modules, {'dispram_recipe': fake}))
        self.enterContext(patch.object(module, 'dispram_guard_present', lambda: True))
        return module

    def test_tp8_renders_eight_ranks(self):
        C = self.cluster(layered_env(8))
        C.validate()
        switches = C.launch_switches()
        self.assertEqual(C.TP, 8)
        self.assertEqual(C.kstop_layout(switches)['blocks'], 16394)
        commands = [C.docker_command(rank, 'glm53full-tp8') for rank in C.RANKS]
        for rank, cmd in enumerate(commands):
            args = cmd[cmd.index(C.IMAGES[rank]) + 1:]
            get = lambda flag: args[args.index(flag) + 1]
            self.assertEqual((get('--tensor-parallel-size'), get('--nnodes'), get('--node-rank')), ('8', '8', str(rank)))
            self.assertEqual(get('--master-addr'), '192.0.2.1')
            self.assertEqual(json.loads(get('--speculative-config'))['draft_tensor_parallel_size'], 8)
            self.assertEqual(json.loads(get('--speculative-config'))['num_speculative_tokens'], 3)
            self.assertEqual(get('--kv-cache-memory-bytes'), str(topology.full_slot_head()))
            self.assertEqual(get('--max-model-len'), '262144')
            self.assertEqual(get('--max-num-seqs'), '4')
            self.assertEqual('--headless' in args, rank != 0)
            self.assertIn('VLLM_HOST_IP=192.0.2.%d' % (rank + 1), cmd)
            self.assertIn('GLM_FP4_POOL_BLOCKS=16394', cmd)
            self.assertFalse(any(t.startswith(('GLM_DRAFT_HEAD=', 'GLM_DRAFT_EHPROJ=', 'GLM_MARLIN_L2HINT=',
                                               'GLM_ATTN_FRONT_FUSE=')) for t in cmd), rank)
        self.assertEqual(len({shlex.join(c) for c in commands}), 8)

    def test_tp4_vector_is_byte_identical_to_saved_one(self):
        C = self.cluster(layered_env(4))
        args = json.loads((ROOT / 'profiles/serve-args.json').read_text())
        self.assertEqual(C.tp_shape(args), args)

    def test_host_count_must_match_tp(self):
        C = self.cluster(layered_env(8, ranks=4))
        with self.assertRaisesRegex(ValueError, 'Exactly 8 hosts'):
            C.validate()
        with patch.dict(C.ENV, NCCL_SHA256=','.join(['b' * 64] * 4)), self.assertRaisesRegex(ValueError, '8 comma'):
            C.nccl_hashes()

    def test_per_host_nccl_hashes(self):
        C = self.cluster(layered_env(8))
        hashes = ['%x' % r * 64 for r in range(1, 9)]
        with patch.dict(C.ENV, NCCL_SHA256=','.join(hashes)):
            self.assertEqual(C.nccl_hashes(), hashes)

    def test_tp4_only_levers_refused_at_tp8(self):
        C = self.cluster(layered_env(8))
        for key, on in (('GLM_DRAFT_HEAD', 'nvfp4'), ('GLM_DRAFT_EHPROJ', 'fp8'), ('GLM_ATTN_FRONT_FUSE', '1'),
                        ('GLM_MARLIN_L2HINT', 'on'), ('GLM_MARLIN_L2HINT_AB', '1'),
                        ('GLM_INDEXER_ROW_SHARD', '1'), ('GLM_INDEXER_STABLE_TOPK', '1')):
            with self.subTest(key=key), patch.dict(C.ENV, {key: on}), self.assertRaisesRegex(ValueError, key + ' is TP4-only'):
                C.tp_gates()
        with patch.dict(C.ENV, RECIPE_PROFILE='dspark-k3'), self.assertRaisesRegex(ValueError, 'native-mtp-k2'):
            C.tp_gates()

    def test_context_capped_at_model_positions(self):
        C = self.cluster(layered_env(8))
        with patch.dict(C.ENV, RECIPE_MAX_MODEL_LEN=str(topology.MODEL['max_position'])):
            C.launch_switches()
        with patch.dict(C.ENV, RECIPE_MAX_MODEL_LEN=str(topology.MODEL['max_position'] + 64)):
            with self.assertRaisesRegex(ValueError, 'max_position_embeddings'):
                C.launch_switches()

    def test_unsupported_tp_size_refused(self):
        with patch.dict(os.environ, dict(layered_env(8), RECIPE_TP_SIZE='16'), clear=True):
            with self.assertRaisesRegex(ValueError, 'RECIPE_TP_SIZE must be one of 4, 8'):
                load_cluster()


class Profile(unittest.TestCase):
    def test_tp8_profile_head_matches_projection(self):
        text = (ROOT / 'profiles/tp8.env').read_text()
        self.assertIn("export RECIPE_KV_HEAD_BYTES='%d'" % topology.full_slot_head(), text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
