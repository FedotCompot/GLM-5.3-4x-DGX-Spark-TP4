# SPDX-License-Identifier: Apache-2.0
"""Offline probe receipts: batched token counting, fail-closed staging and errors."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe', ROOT/'scripts/decode_fair_staged_probe.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def sse(tokens=(2, 3), usage=5, done=True):
    rows = [dict(choices=[dict(text='x', logprobs={'tokens': ['x']*n})]) for n in tokens]
    if usage is not None: rows.append(dict(choices=[], usage={'completion_tokens': usage}))
    data = b''.join(b'data: '+json.dumps(row).encode()+b'\n\n' for row in rows)
    if done: data += b'data: [DONE]\n\n'
    return io.BytesIO(data)


class Probe(unittest.TestCase):
    def test_mtp_batches_count_tokens_not_sse_messages(self):
        with patch.object(P, 'urlopen', return_value=sse()) as http, \
             patch.object(P.time, 'monotonic', side_effect=[10., 12., 14.]):
            result = P.stream('http://127.0.0.1:8095/v1', 'model', [1, 2], 5, 3)
        self.assertEqual(result['ttft_s'], 2.)
        self.assertEqual(result['decode_tok_s'], 1.5)
        self.assertEqual(result['events'], [(12., 2), (14., 3)])
        payload = json.loads(http.call_args.args[0].data)
        self.assertTrue(payload['ignore_eos'])
        self.assertEqual(payload['logprobs'], 1)

    def test_incomplete_missing_usage_or_count_mismatch_fail(self):
        for response in (sse(done=False), sse(usage=None), sse(usage=6), sse(tokens=(), usage=0)):
            with patch.object(P, 'urlopen', return_value=response), self.assertRaises(RuntimeError):
                P.stream('http://127.0.0.1:8095/v1', 'model', [1], 5, 3)

    def test_valid_staging_exact_overlap_rate_and_invalid_staging(self):
        # No HTTP call: concurrent submission uses independently labeled receipts.
        args = NS(prefix_tokens=NS(read_text=lambda: '[1, 2, 3]'), running_token=4,
                  first_token=5, queued_token=6, residual_tokens=256, timeout=1,
                  decode_tokens=1024, prefill_output_tokens=32, base='http://127.0.0.1:8095/v1',
                  model='model', running_delay=0, queue_delay=0, label='test')
        salts = []
        def receipt(*_args, **kwargs):
            salts.append(kwargs['cache_salt'])
            label = kwargs.get('label') or _args[-1]
            if label == 'warm': start, first, last = 0., 1., 2.
            if label == 'A-running':
                start, first, last = 3., 4., 20.
                _args[5].set()
            if label == 'B-first-prefill':
                start, first, last = 5., 7., 10.
                _args[6].set()
            if label == 'C-queued-prefill': start, first, last = 6., 11., 12.
            events = [(6., 2), (8., 3), (10., 2)] if label == 'A-running' else [(first, 1)]
            return dict(label=label, started=start, first=first, last=last, events=events,
                        usage={}, ttft_s=first-start, decode_tok_s=2.)
        with patch.object(P, 'stream', side_effect=receipt): result = P.run(args)
        self.assertEqual(len(set(salts)), 1)
        self.assertTrue(salts[0].startswith('decode-fair-probe-'))
        self.assertAlmostEqual(result['running_overlap_tok_s'], 7/6)
        self.assertEqual(result['queued_ttft_s'], 5.)
        def invalid(*a, **kw):
            row = receipt(*a, **kw)
            if row['label'] == 'C-queued-prefill': row['started'] = 8.
            return row
        with patch.object(P, 'stream', side_effect=invalid), self.assertRaisesRegex(RuntimeError, 'invalid staging'):
            P.run(args)


if __name__ == '__main__': unittest.main()
