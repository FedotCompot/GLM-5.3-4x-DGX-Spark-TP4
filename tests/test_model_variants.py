"""Stock and abliterated checkpoints: pinned manifests, the RECIPE_MODEL switch and sidecar provenance."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import weights
from test_launcher_tp8 import ClusterCase, layered_env

ABLITERATED = dict(repo='Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix', sha='08a92f9da01cd90d591edb3fdd71c2ccc9f37ddd')
SHARED = ('LICENSE', 'chat_template.jinja', 'config.json', 'generation_config.json', 'model.safetensors.index.json',
          'tokenizer.json', 'tokenizer_config.json')
MTP_SHARDS = ['model-%05d-of-00282.safetensors' % i for i in range(270, 275)]
SITE = ('ABLITERATED_MODEL_DIR=/models/abliterated\nABLITERATED_ATTN_NVFP4_DIR=/srv/abl-attn\n'
        'ABLITERATED_NVFP4_MORE_DIR=/srv/abl-more\n')


def manifest(kind):
    return weights.manifest_for(kind)


def files(kind):
    return {f['f']: f for f in manifest(kind)['files']}


def sidecar_from(kind, shards):
    shas = files(kind)
    return dict(source=dict(index_sha256=shas['model.safetensors.index.json']['sha256'],
                            config_sha256=shas['config.json']['sha256'],
                            shards={name: shas[name]['sha256'] for name in shards}))


class Manifests(unittest.TestCase):
    def test_abliterated_manifest_is_complete_and_pinned(self):
        m = manifest('abliterated')
        weights.validate_manifest(m)
        self.assertEqual((m['repo'], m['sha'], m['schema']), (ABLITERATED['repo'], ABLITERATED['sha'], 1))
        self.assertEqual(sum(f['f'].endswith('.safetensors') for f in m['files']), 282)
        text = ''.join('%s  %s\n' % (f['sha256'], f['f']) for f in m['files'])
        self.assertEqual(text, (ROOT / 'manifests/abliterated.sha256').read_text())

    def test_same_layout_as_stock_with_different_weights(self):
        stock, abl = files('target'), files('abliterated')
        self.assertEqual(set(stock), set(abl))
        for name in SHARED:
            self.assertEqual(stock[name], abl[name], name)
        changed = [n for n in stock if n.endswith('.safetensors') and stock[n]['sha256'] != abl[n]['sha256']]
        self.assertEqual(len(changed), 278)
        self.assertEqual([n for n in MTP_SHARDS if n not in changed], MTP_SHARDS[1:4])

    def test_cli_kinds(self):
        self.assertEqual(weights.KINDS, ('target', 'abliterated', 'drafter'))
        with self.assertRaises(ValueError):
            weights.manifest_for('derisked')


class Builder(unittest.TestCase):
    def opener(self, siblings, bodies, sha='a' * 40):
        def fake(url, timeout=None):
            if '/api/models/' in url:
                return io.BytesIO(json.dumps(dict(sha=sha, siblings=siblings)).encode())
            return io.BytesIO(bodies[url.rsplit('/', 1)[-1]])
        return fake

    def test_lfs_hashes_from_hub_small_files_hashed_locally(self):
        siblings = [dict(rfilename='model-1.safetensors', lfs=dict(sha256='b' * 64, size=10)),
                    dict(rfilename='config.json'), dict(rfilename='.gitattributes'),
                    dict(rfilename='chat_template.jinja.bak-prethinking')]
        built = weights.build_manifest('org/repo', 'a' * 40, self.opener(siblings, {'config.json': b'{}'}))
        self.assertEqual(built['files'], [
            dict(f='config.json', size=2, sha256='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'),
            dict(f='model-1.safetensors', size=10, sha256='b' * 64)])

    def test_refuses_moved_revision(self):
        with self.assertRaisesRegex(ValueError, 'different revision'):
            weights.build_manifest('org/repo', 'a' * 40, self.opener([], {}, sha='c' * 40))


class Switch(ClusterCase):
    def test_stock_is_default(self):
        env = layered_env(4, SITE)
        self.assertEqual((env['RECIPE_MODEL'], env['MODEL_DIR'], env['GLM_ATTN_NVFP4_DIR']),
                         ('stock', '/models/target', '/srv/attn'))
        C = self.cluster(env)
        self.assertEqual(C.model_kind(), 'target')
        self.assertIn('Tech2wild/GLM-5.3-Int4-Int8Mix@206507bbb047d8223964a0414cd83230c59428f9', C.model_note())

    def test_abliterated_swaps_checkpoint_and_sidecars(self):
        envs = {tp: layered_env(tp, SITE + 'RECIPE_MODEL=abliterated\n') for tp in (4, 8)}
        for tp, env in envs.items():
            self.assertEqual((env['MODEL_DIR'], env['GLM_ATTN_NVFP4_DIR'], env['GLM_NVFP4_MORE_DIR']),
                             ('/models/abliterated', '/srv/abl-attn', '/srv/abl-more'))
            C = self.cluster(env)
            self.assertEqual(C.model_kind(), 'abliterated')
            self.assertEqual(C.sidecar_dirs(), ['/srv/abl-attn', '/srv/abl-more'])
            cmd = C.docker_command(0, 'glm53full-abl')
            self.assertIn('/models/abliterated:/model:ro', cmd)
            self.assertIn('/srv/abl-attn:/attn-nvfp4:ro', cmd)
            self.assertIn('/srv/abl-more:/more-nvfp4:ro', cmd)
            args = cmd[cmd.index(C.IMAGES[0]) + 1:]
            self.assertEqual(args[args.index('--served-model-name') + 1], 'GLM-5.3')

    def test_abliterated_needs_its_own_paths(self):
        with self.assertRaisesRegex(RuntimeError, 'ABLITERATED_MODEL_DIR'):
            layered_env(4, 'RECIPE_MODEL=abliterated\n')
        with self.assertRaisesRegex(RuntimeError, 'stock or abliterated'):
            layered_env(4, 'RECIPE_MODEL=derisked\n')

    def test_model_ids_match_manifests(self):
        C = self.cluster(layered_env(4))
        for kind in C.MODEL_MANIFESTS.values():
            m = manifest(kind)
            self.assertEqual(C.MODEL_IDS[kind], m['repo'] + '@' + m['sha'])

    def test_launcher_refuses_unknown_model(self):
        C = self.cluster(layered_env(4))
        with patch.dict(C.ENV, RECIPE_MODEL='uncensored'), self.assertRaisesRegex(ValueError, 'RECIPE_MODEL'):
            C.model_kind()


class SidecarProvenance(ClusterCase):
    def test_stock_sidecars_refused_for_abliterated_checkpoint(self):
        C = self.cluster(layered_env(4))
        abl = manifest('abliterated')
        attn_shards = ['model-%05d-of-00282.safetensors' % i for i in range(2, 84)]
        self.assertEqual(C.sidecar_problems(sidecar_from('abliterated', attn_shards), abl), [])
        stale = C.sidecar_problems(sidecar_from('target', attn_shards), abl)
        self.assertEqual(len(stale), len(attn_shards))
        mtp = C.sidecar_problems(sidecar_from('target', MTP_SHARDS), abl)
        self.assertEqual(mtp, [MTP_SHARDS[0], MTP_SHARDS[4]])
        self.assertEqual(C.sidecar_problems(sidecar_from('target', attn_shards), manifest('target')), [])

    def test_sidecar_without_shard_record_or_wrong_index_refused(self):
        C = self.cluster(layered_env(4))
        stock = manifest('target')
        self.assertEqual(C.sidecar_problems(dict(source={}), stock)[0], 'no recorded source shards')
        bad = sidecar_from('target', MTP_SHARDS)
        bad['source']['index_sha256'] = '0' * 64
        self.assertEqual(C.sidecar_problems(bad, stock), ['model.safetensors.index.json'])

    def test_preflight_checks_every_mounted_sidecar(self):
        C = self.cluster(layered_env(4))
        with patch.dict(C.ENV, GLM_ATTN_WEIGHTS='int8'):
            self.assertEqual(C.sidecar_dirs(), [])
        with patch.dict(C.ENV, GLM_NVFP4_GROUPS='attn'):
            self.assertEqual(C.sidecar_dirs(), ['/srv/attn'])
        source = (ROOT / 'scripts/cluster.py').read_text()
        self.assertIn("for path in sidecar_dirs():", source)
        self.assertIn("checkpoints[0][0])", source)


class BootPreflight(unittest.TestCase):
    def test_model_option_reaches_worker(self):
        text = (ROOT / 'scripts/boot_preflight.py').read_text()
        self.assertIn("'--model',a.model]", text)
        self.assertIn("ROOT/'manifests'/(a.model+'.json')", text)
        out = subprocess.run([sys.executable, str(ROOT / 'scripts/boot_preflight.py'), '--model', 'derisked',
                              '--out', os.devnull], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("invalid choice: 'derisked'", out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
