#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded staged-arrival SSE probe; run only on the coordinator's diagnostic boot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid
from urllib.request import Request, urlopen


def stream(base, model, prompt, limit, timeout, first=None, submitted=None, label='', cache_salt=None):
    payload = dict(model=model, prompt=prompt, max_tokens=limit, temperature=0,
                   ignore_eos=True, stream=True, stream_options={'include_usage': True}, logprobs=1)
    if cache_salt is not None: payload['cache_salt'] = cache_salt
    started = time.monotonic()
    if submitted is not None: submitted.set()
    req = Request(base.rstrip('/') + '/completions', data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
    events, usage, done = [], None, False
    with urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            item = json.loads(data)
            if item.get('error'): raise RuntimeError('server stream error')
            if item.get('usage'): usage = item['usage']
            for choice in item.get('choices', []):
                tokens = (choice.get('logprobs') or {}).get('tokens') or []
                if tokens:
                    events.append((time.monotonic(), len(tokens)))
                    if first is not None: first.set()
    if not done or not events or usage is None:
        raise RuntimeError('incomplete stream, missing token logprobs or missing usage')
    total = usage.get('completion_tokens')
    counted = sum(n for _, n in events)
    if total != counted:
        raise RuntimeError('stream logprob token count does not match completion usage')
    first_at, last_at = events[0][0], events[-1][0]
    duration = last_at - first_at
    # Exclude first emitted group from first-to-last decode rate.
    return dict(label=label, started=started, first=first_at, last=last_at,
                ttft_s=first_at-started, usage=usage, events=events,
                decode_tok_s=(total-events[0][1])/duration if duration > 0 else None)


def run(a):
    prefix = json.loads(a.prefix_tokens.read_text())
    if not isinstance(prefix, list) or not prefix or any(type(t) is not int or t < 0 for t in prefix):
        raise ValueError('prefix file must be a nonempty JSON array of token IDs')
    if len({a.running_token, a.first_token, a.queued_token}) != 3:
        raise ValueError('three distinct suffix tokens required to avoid cross-request APC')
    suffix = lambda token: prefix + [token] * a.residual_tokens
    # Warm only the common prefix. Cache hits must be verified from receipts;
    # warming does not guarantee retained cache under memory pressure.
    # Fresh per-trial namespace avoids fully cached B/C tails from earlier arms.
    salt = 'decode-fair-probe-' + uuid.uuid4().hex
    warm = stream(a.base, a.model, prefix, 1, a.timeout, label='warm', cache_salt=salt)
    ready_a, sent_b = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fa = pool.submit(stream, a.base, a.model, suffix(a.running_token), a.decode_tokens,
                         a.timeout, ready_a, None, 'A-running', cache_salt=salt)
        if not ready_a.wait(a.timeout): raise RuntimeError('A did not begin decoding')
        time.sleep(a.running_delay)
        fb = pool.submit(stream, a.base, a.model, suffix(a.first_token), a.prefill_output_tokens,
                         a.timeout, None, sent_b, 'B-first-prefill', cache_salt=salt)
        if not sent_b.wait(a.timeout): raise RuntimeError('B did not submit')
        # C arrives during B's mixed step; arrival validity is checked below.
        time.sleep(a.queue_delay)
        fc = pool.submit(stream, a.base, a.model, suffix(a.queued_token), a.prefill_output_tokens,
                         a.timeout, None, None, 'C-queued-prefill', cache_salt=salt)
        running, b, c = fa.result(), fb.result(), fc.result()
    begin, end = b['started'], c['first']
    if not (running['first'] < begin < c['started'] < b['first'] <= end < running['last']):
        raise RuntimeError('invalid staging: C missed B prefill or A stopped before C TTFT')
    overlap = [(t, n) for t, n in running['events'] if begin <= t <= end]
    if len(overlap) < 2: raise RuntimeError('too few running decode emissions in overlap')
    gaps = [right[0]-left[0] for left, right in zip(overlap, overlap[1:])]
    # Exact token count over wall time from B submission through C first token.
    rate = sum(n for _, n in overlap)/(end-begin)
    epoch = running['started']
    for result in (warm, running, b, c):
        result['started_s'] = result.pop('started') - epoch
        result['first_s'] = result.pop('first') - epoch
        result['last_s'] = result.pop('last') - epoch
        result['events'] = [[t-epoch, n] for t, n in result['events']]
    return dict(kind='MEASURED', label=a.label, cache_salt=salt, prefix_tokens=len(prefix), residual_tokens=a.residual_tokens,
                queued_ttft_s=c['ttft_s'], running_overlap_tok_s=rate, running_overlap_max_gap_s=max(gaps),
                running_decode_tok_s=running['decode_tok_s'], staging_valid=True,
                cache_note='inspect usage.prompt_tokens_details.cached_tokens; absent means unknown',
                requests=[warm, running, b, c])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8095/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--prefix-tokens', required=True, type=Path)
    for name in ('running', 'first', 'queued'):
        p.add_argument('--'+name+'-token', required=True, type=int)
    p.add_argument('--residual-tokens', type=int, default=256)
    p.add_argument('--decode-tokens', type=int, default=1024)
    p.add_argument('--prefill-output-tokens', type=int, default=32)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--running-delay', type=float, default=1.0)
    p.add_argument('--queue-delay', type=float, default=.050)
    p.add_argument('--label', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if min(a.residual_tokens, a.decode_tokens, a.prefill_output_tokens, a.timeout) <= 0:
        p.error('token limits and timeout must be positive')
    if min(a.running_delay, a.queue_delay, a.running_token, a.first_token, a.queued_token) < 0:
        p.error('delays and token IDs must be nonnegative')
    result = run(a)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}, sort_keys=True))


if __name__ == '__main__': main()
