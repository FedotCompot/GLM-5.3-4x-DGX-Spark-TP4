# SPDX-License-Identifier: Apache-2.0
"""Load both real registrations with toolkit libcuda stub; forbid CUDA init."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch


def check_schemas(stock, hint):
    """Used in both no-GPU build checks and every enabled serving boot."""
    a, b = stock.default._schema, hint.default._schema
    if len(a.arguments) != 29 or len(b.arguments) != 30:
        raise RuntimeError('schema argument count drift')
    expected = str(a).replace('_moe_C::', '_glm_l2hint_C::').replace(
        ') ->', ', bool l2_hint=False) ->')
    if str(b) != expected:
        raise RuntimeError('candidate ABI differs from stock beyond hint bool')
    return str(a), str(b)


def main(out):
    import importlib.util
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('no-GPU selfcheck requires uninitialized CUDA')
    def forbidden(*a, **kw):raise RuntimeError('CUDA init forbidden in selfcheck')
    receipt=json.loads((out/'candidate/build.json').read_text())
    root=Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    with patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.cuda,'init',forbidden):
        torch.ops.load_library(str(root/receipt['stock_binary_name']))
        torch.ops.load_library(str(out/'candidate/marlin.so'))
        stock=torch.ops._moe_C.moe_wna16_marlin_gemm
        hint=torch.ops._glm_l2hint_C.moe_wna16_marlin_gemm
        a, b = check_schemas(stock, hint)
        t=torch.empty((1,),dtype=torch.bfloat16)
        argv=[t,None,t,None,t,None,None,None,None,None,t,t,t,t,t,
              8,8,False,0,4,1024,6144,True,False,True,False,-1,-1,-1]
        for op,values in ((stock,argv),(hint,argv+[False]),(hint,argv+[True])):
            try:op(*values)
            except NotImplementedError as exc:
                if 'CPU' not in str(exc):raise
            else:raise RuntimeError('unexpected CPU kernel registered')
        if torch.cuda.is_initialized():raise RuntimeError('selfcheck initialized CUDA')
    (out/'selfcheck.json').write_text(json.dumps(dict(status='SELFCHECK_PASS',cuda_initialized=False,
        stock_schema=str(a),hint_schema=str(b)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
