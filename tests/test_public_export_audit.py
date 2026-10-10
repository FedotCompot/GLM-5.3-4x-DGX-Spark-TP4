# SPDX-License-Identifier: Apache-2.0
"""Export regressions: private paths/endpoints, scanner contents and outgoing history."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from public_export_audit import audit, audit_history


class PublicExport(unittest.TestCase):
    def test_each_private_location_fails_cli_without_disclosing_bytes(self):
        # Assemble synthetic fixtures so the tests themselves remain exportable.
        examples = [
            ('/'+'var/folders/xx/synthetic/T/receipt', 'temp_path'),
            ('/'.join(('', 'private', 'var', 'folders', 'xx', 'synthetic', 'T', 'receipt')), 'temp_path'),
            ('/'+'private/tmp/receipt', 'temp_path'),
            ('/'+'tmp/'+'clau'+'de-session/receipt', 'temp_path'),
            ('/'+'Users/synthetic/receipt', 'home_path'),
            ('/'+'home/synthetic/receipt', 'home_path'),
            ('synthetic.'+'ts.net', 'tailnet'),
            ('.'.join(('100', '64', '12', '34')), 'tailnet_ip'),
            ('.'.join(('100', '1', '2', '3')), 'tailnet_ip'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            path = root/'receipt.txt'
            path.touch()
            subprocess.run(['git', '-C', tmp, 'add', path.name], check=True)
            for value, kind in examples:
                with self.subTest(kind=kind, value=value):
                    path.write_text(value+'\n')
                    result = subprocess.run([sys.executable, str(ROOT/'scripts/public_export_audit.py'),
                                             '--root', tmp], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1)
                    report = json.loads(result.stdout)
                    self.assertFalse(report['passed'])
                    self.assertIn(kind, [f['kind'] for f in report['findings']])
                    self.assertNotIn(value, result.stdout)

    def test_scanner_source_is_scanned_and_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', tmp], check=True)
            path = root/'scripts/public_export_audit.py'
            path.parent.mkdir()
            path.write_bytes((ROOT/'scripts/public_export_audit.py').read_bytes())
            subprocess.run(['git', '-C', tmp, 'add', '.'], check=True)
            self.assertTrue(audit(root)['passed'])
            with path.open('a') as stream:
                stream.write('\n# '+ '/'+'Users/synthetic/receipt\n')
            self.assertFalse(audit(root)['passed'])

    def test_outgoing_history_catches_deleted_leak_and_commit_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', '-C', tmp, *args], text=True).strip()
            git('init', '-q')
            git('config', 'user.name', 'knapcio')
            git('config', 'user.email', 'knapcio@gmail.com')
            (root/'clean.txt').write_text('portable receipt\n')
            git('add', '.')
            git('commit', '-qm', 'Clean public base')
            base = git('rev-parse', 'HEAD')
            leak = root/'receipt.txt'
            leak.write_text('/'+'private/tmp/receipt\n')
            git('add', '.')
            git('commit', '-qm', 'Add receipt')
            leak.unlink()
            git('add', '-u')
            git('commit', '-qm', 'Remove receipt\n\nCo-authored-by: '+'Clau'+'de')
            self.assertTrue(audit(root)['passed'])
            report = audit_history(root, base+'..HEAD')
            self.assertFalse(report['passed'])
            self.assertEqual(len(report['commits']), 2)
            self.assertEqual({f['kind'] for f in report['findings']},
                             {'temp_path', 'authorship_or_session'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
