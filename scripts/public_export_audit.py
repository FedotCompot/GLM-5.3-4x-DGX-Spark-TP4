#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit release trees and outgoing commits; report locations, never secret bytes."""
import argparse
import collections
import ipaddress
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'home_path': r'\x2f(?:Users|home)\x2f',
    'temp_path': r'\x2f(?:var\x2ffolders|private\x2fvar\x2ffolders|private\x2ftmp)(?=\x2f|\b)|\x2ftmp\x2f\x63\x6c\x61\x75\x64\x65[^\s"\x27<>]*',
    'site_hostname': r'(?i)\b(?:spark[-_]?0|\x6euc-agent)\b',
    'site_subnet': r'(?<![\w.])10\.100\.(?:96|97)(?:\.(?:[0-9]{1,3}|x))?(?![\w.])',
    'tailnet': r'(?i)[\w.-]+\.ts\.net|tail[0-9a-f]{4,}',
    'email': r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
    'authorship_or_session': r'(?i)\b(?:\x63\x6c\x61\x75\x64\x65|\x63\x68\x61\x74\x67\x70\x74|\x67\x70\x74-[0-9][\w.-]*|\x63\x6f\x64\x65\x78|\x61\x73\x74\x72\x61|\x6f\x70\x75\x73)\b|https?://[^\s"\x27]*(?:session|\x63\x68\x61\x74\x67\x70\x74\.com|\x63\x6c\x61\x75\x64\x65\.ai)[^\s"\x27]*',
    'key_material': r'-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----|\b(?:ghp_|github_pat_|sk-[A-Za-z0-9]|tskey-)[A-Za-z0-9_-]+',
}
IP = re.compile(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])')


def scan(name, text, findings, allowed, commit=None):
    for number,line in enumerate(text.splitlines(),1):
        location = dict(file=name, line=number)
        if commit is not None:
            location['commit'] = commit
        for kind,pattern in PATTERNS.items():
            for match in re.finditer(pattern,line):
                if kind=='email' and match[0]=='knapcio@gmail.com':
                    allowed['owner_email']+=1;continue
                findings.append(dict(location,kind=kind))
        for match in IP.finditer(line):
            try: ip=ipaddress.IPv4Address(match[0])
            except ipaddress.AddressValueError:
                allowed['invalid_ipv4_test_fixture']+=1;continue
            if ip.is_loopback or ip in ipaddress.IPv4Network('192.0.2.0/24'):
                allowed['loopback_or_TEST_NET_fixture']+=1
            else:
                kind = 'tailnet_ip' if match[0].split('.')[0] == '100' else 'site_ip'
                findings.append(dict(location,kind=kind))


def audit(root):
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    findings=[]; allowed=collections.Counter(); files=0
    for name in tracked:
        path=root/name
        if not path.is_file(): continue
        try: text=path.read_text()
        except UnicodeError: continue
        files+=1
        scan(name, text, findings, allowed)
        scan(name, name, findings, allowed)
    return dict(scope='tracked working tree',text_files=files,
                findings=findings,allowed=dict(allowed),passed=not findings)


def audit_history(root, revision_range):
    """Scan metadata and every full tree in the exact outgoing range, including deleted files."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    commits = git('rev-list', '--reverse', revision_range).decode().splitlines()
    findings=[]; allowed=collections.Counter(); files=0
    for commit in commits:
        scan('<commit metadata>', git('show', '-s', '--format=raw', commit).decode(),
             findings, allowed, commit)
        for entry in git('ls-tree', '-r', '-z', commit).split(b'\0'):
            if not entry: continue
            info, name = entry.split(b'\t', 1)
            _, kind, oid = info.split()
            name = name.decode()
            scan(name, name, findings, allowed, commit)
            if kind != b'blob':
                findings.append(dict(file=name, line=0, kind='unscanned_gitlink', commit=commit))
                continue
            raw = git('cat-file', 'blob', oid.decode())
            if b'\0' in raw: continue
            try: text = raw.decode()
            except UnicodeError: continue
            files += 1
            scan(name, text, findings, allowed, commit)
    return dict(scope=revision_range, commits=commits, text_files=files,
                findings=findings, allowed=dict(allowed), passed=not findings)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--history',help='exact outgoing revision range; scan each full tree and commit metadata')
    a=p.parse_args();result=audit(a.root)
    if a.history:
        result['history'] = audit_history(a.root, a.history)
        result['passed'] = result['passed'] and result['history']['passed']
    text=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.write_text(text)
    print(text,end='')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
