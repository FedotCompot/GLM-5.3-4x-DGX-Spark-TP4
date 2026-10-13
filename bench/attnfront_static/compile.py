# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'overlay/bringup'))
from attnfront_native.build import build,verify
out=Path(sys.argv[1]);native=out/'native';verify(native)
import torch
assert not torch.cuda.is_initialized()
import shutil
package_native=ROOT/'overlay/bringup/attnfront_native/build'
package_native.mkdir(exist_ok=True)
for name in ('attnfront.so','build.json'):
    shutil.copy2(native/name,package_native/name)
import glm_attnfront as a
a.CONFIG=(True,True,True)
a.register_ops()
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    for m in range(1,17):
        q=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16)
        k=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
        torch.ops.glm_attnfront.dual_rms(q,k,torch.empty(1536,device='cuda',dtype=q.dtype),
             torch.empty(512,device='cuda',dtype=q.dtype),1e-6,1e-5,torch.empty_like(q),torch.empty_like(k))
        r=torch.empty((m,16,64),device='cuda',dtype=q.dtype)
        kr=torch.empty((m,1,64),device='cuda',dtype=q.dtype)
        dep=torch.ops.glm_attnfront.rope_update(r,kr,k,torch.empty(m,device='cuda',dtype=torch.int64),
            torch.empty((4096,64),device='cuda',dtype=q.dtype),'fake-layer',False)
        assert dep.shape==(0,) and dep.dtype==q.dtype
assert not torch.cuda.is_initialized()
(out/'selfcheck.json').write_text(json.dumps(dict(schema=str(torch.ops.glm_attnfront.dual_rms.default._schema),
    fake_m=list(range(1,17)),cuda_initialized=torch.cuda.is_initialized()),indent=2)+'\n')
