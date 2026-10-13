# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
import attnfront_crossboot as x

class Panel(unittest.TestCase):
    def teacher(self,v=-.5):return [dict(id='short',tokens=[1,2],prompt_lp=[None,{'2':v}])]
    def panels(self,tmp):
        paths=[]
        for phase,mode in (('A','off'),('B','on'),('Aprime','off')):
            rows=[]
            for kind,count in (('prose',15),('code',10)):
                for i in range(count):
                    ms=9.5 if phase=='B' else 10
                    metrics=dict(status='OK',request_id=f'{phase}-{kind}-{i}',metadata=dict(pair_key=f'{kind}-{i}',prompt_sha256=f'prompt-{kind}-{i}',seed=0,k_mode='boot-default',target_M='boot-default'),time_source='client_first_to_last_output_token',labels={'model_name':'GLM-5.3'},cycle_ms=ms,committed_per_cycle=3,cycle_ms_kind='request wall')
                    rows.append(dict(kind=kind,i=i,cycle_ms=ms,token_ids=[1,2],text='ok',finish_reason='stop',completion_tokens=2,metrics=metrics))
            path=Path(tmp)/(phase+'.json');path.write_text(json.dumps(dict(status='COMPLETE',mode=mode,boot=phase,rows=rows,teacher=[self.teacher(),self.teacher()])))
            paths.append(path)
        return paths
    def test_exact_15_10_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=x.compare(self.panels(tmp))
            self.assertEqual(result['verdict'],'KEEP')
            self.assertTrue(result['teacher_bit_identity'])
            self.assertEqual(result['panel']['prose']['n'],15)
    def test_token_or_teacher_mismatch_kills(self):
        for kind in ('token','teacher'):
            with tempfile.TemporaryDirectory() as tmp:
                paths=self.panels(tmp);p=paths[1];v=json.loads(p.read_text())
                if kind=='token':v['rows'][0]['token_ids']=[1,3]
                else:v['teacher']=[self.teacher(-.5000000000000001)]*2
                p.write_text(json.dumps(v))
                with self.assertRaises(ValueError):x.compare(paths)
    def test_zero_sign_bits_and_incomplete_panel(self):
        self.assertNotEqual(x.teacher_bits(self.teacher(0.)),x.teacher_bits(self.teacher(-0.)))
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.panels(tmp);p=paths[0];v=json.loads(p.read_text());v['rows'].pop();p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):x.compare(paths)
    def test_gain_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.panels(tmp);p=paths[1];v=json.loads(p.read_text())
            for r in v['rows']:r['cycle_ms']=r['metrics']['cycle_ms']=9.99
            p.write_text(json.dumps(v));self.assertEqual(x.compare(paths)['verdict'],'KILL')

if __name__=='__main__':unittest.main()
