# SPDX-License-Identifier: Apache-2.0
import hashlib,json,os
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='attnfront_static' else Path(__file__).parent
sys.path.insert(0,str(ROOT/'overlay/bringup'))
import glm_attnfront as a
class Static(unittest.TestCase):
    def test_pins(self):
        image=Path(os.environ['GLM_IMAGE_SRC'])/'vllm'
        for name,pin in a.PINS.items():
            self.assertEqual(hashlib.sha256((image/(name.removeprefix('vllm.').replace('.','/')+'.py')).read_bytes()).hexdigest(),pin)
    def test_transforms(self):
        image=Path(os.environ['GLM_IMAGE_SRC'])/'vllm'
        for name,fn in ((a.FRONT,a.transform_front),(a.COMMON,a.transform_common)):
            compile(fn((image/(name.removeprefix('vllm.').replace('.','/')+'.py')).read_text()),name,'exec')
    def test_vendor(self):
        base=ROOT/'overlay/bringup/attnfront_native/vendor'
        for name,pin in json.loads((base/'BASE.json').read_text())['files'].items():
            self.assertEqual(hashlib.sha256((base/Path(name).name).read_bytes()).hexdigest(),pin)
    def test_disabled(self):
        self.assertFalse(a.register({}))
if __name__=='__main__':unittest.main()
