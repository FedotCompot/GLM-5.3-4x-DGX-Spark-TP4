import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dispram_recipe as D


class DispramHome(unittest.TestCase):
    def calls(self, env, fn):
        seen = []
        def run(host, cmd, timeout=120):
            seen.append(cmd)
            return 0, 'ok'
        fn(run)
        return seen

    def test_site_home_is_passed_to_every_remote_call(self):
        env = {'DISPRAM_HOME': '/opt/site/dispram', 'RECIPE_DISPRAM': 'require'}
        seen = self.calls(env, lambda run: D.ensure_node('h', env, run=run, sleep=lambda s: None))
        seen += self.calls(env, lambda run: D.postcheck(['h'], env, run=run))
        self.assertEqual(sum('dispram.sh' in c for c in seen), 4)  # invalid-state, watch, preborrow, postcheck
        for cmd in seen:
            if 'dispram.sh' in cmd:
                self.assertIn('DISPRAM_HOME=/opt/site/dispram /opt/site/dispram/dispram.sh', cmd)

    def test_oom_filter_flag_kept(self):
        env = {'DISPRAM_HOME': '/d', 'DISPRAM_ALLOW_RM_ALLOC_OOM': '1'}
        self.assertEqual(D.script(env), 'DISPRAM_ALLOW_RM_ALLOC_OOM=1 DISPRAM_HOME=/d /d/dispram.sh')

    def test_default_home_matches_recipe(self):
        self.assertEqual(D.script({}), 'DISPRAM_HOME=' + D.HOME + ' ' + D.HOME + '/dispram.sh')


if __name__ == '__main__':
    unittest.main()
