#!/usr/bin/env python3
"""Download or verify pinned checkpoints using complete SHA256 manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

BLOCK = 8 << 20
RECEIPT_MAX_AGE_S = 7 * 24 * 3600   # a full rehash at least once a week


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(BLOCK):
            h.update(block)
    return h.hexdigest()


def validate_manifest(manifest):
    if len(manifest['sha']) != 40:
        raise ValueError('Full immutable revision required')
    seen = set()
    for entry in manifest['files']:
        path = Path(entry['f'])
        if path.is_absolute() or '..' in path.parts or entry['f'] in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(entry['f'])
        if not isinstance(entry['size'], int) or entry['size'] < 0 or len(entry['sha256']) != 64:
            raise ValueError('Every file needs size and SHA256')
        int(entry['sha256'], 16)


def identity(path):
    # Any write, rename, replacement or metadata change moves ctime/mtime/inode.
    st = path.stat()
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev]


def receipt_path(root, manifest):
    key = hashlib.sha256(json.dumps([str(root.resolve()), manifest], sort_keys=True).encode()).hexdigest()
    return Path.home() / '.cache/glm53-full-verify' / (key + '.json')


def verify(root, manifest, receipt=None, now=None):
    """Full SHA256 of every file. With a receipt path, skip hashing only when the last
    full pass for this exact manifest and directory is less than RECEIPT_MAX_AGE_S old
    and every file's (size, mtime, ctime, inode, device) still equals it; otherwise hash
    everything and rewrite the receipt."""
    validate_manifest(manifest)
    now = time.time() if now is None else now
    if receipt is not None and receipt.is_file():
        try:
            saved = json.loads(receipt.read_text())
            age = now - float(saved['written'])
            if 0 <= age < RECEIPT_MAX_AGE_S and saved['files'] == {e['f']: identity(root / e['f']) for e in manifest['files']}:
                print('SHA256 CACHED: %s, %d files unchanged since the full pass %.1f h ago'
                      % (manifest['repo'], len(manifest['files']), age / 3600))
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
    ids = {}
    for entry in manifest['files']:
        path = root / entry['f']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Missing or wrong size: ' + str(path))
        before = identity(path)
        if digest(path) != entry['sha256']:
            raise ValueError('SHA256 mismatch: ' + str(path))
        if identity(path) != before:
            raise ValueError('File changed while hashing: ' + str(path))
        ids[entry['f']] = before
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt.with_name(receipt.name + '.tmp')
        tmp.write_text(json.dumps(dict(written=now, files=ids), sort_keys=True))
        os.replace(tmp, receipt)
    print('SHA256 PASS: %s, %d files' % (manifest['repo'], len(manifest['files'])))


def download(root, manifest):
    validate_manifest(manifest)
    root.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        path = root / entry['f']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size == entry['size'] and digest(path) == entry['sha256']:
                print('verified', entry['f'], flush=True)
                continue
            raise ValueError('Existing file does not match; preserve and inspect: ' + str(path))
        part = path.with_name(path.name + '.part')
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (
            manifest['repo'], manifest['sha'], urllib.parse.quote(entry['f'], safe='/'))
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry['size']:
                raise ValueError('Oversized partial file: ' + str(part))
            if offset == entry['size']:
                break
            headers = {'Range': 'bytes=%d-' % offset} if offset else {}
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as src:
                    if offset and src.status != 206:
                        raise ValueError('Server did not honor resume range for ' + entry['f'])
                    with part.open('ab' if offset else 'wb') as dst:
                        while block := src.read(BLOCK):
                            dst.write(block)
                        dst.flush()
                        os.fsync(dst.fileno())
                break
            except (OSError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if part.stat().st_size != entry['size'] or digest(part) != entry['sha256']:
            raise ValueError('Downloaded file fails size/SHA256; partial preserved: ' + str(part))
        part.rename(path)
        print('downloaded and verified', entry['f'], flush=True)
    verify(root, manifest)


# Hub metadata, not serving assets; the target also drops its local template backup.
MANIFEST_EXCLUDE = ('.gitattributes', 'chat_template.jinja.bak-prethinking')
KINDS = ('target', 'abliterated', 'drafter')


def hub_files(repo, sha, opener=urllib.request.urlopen):
    url = 'https://huggingface.co/api/models/%s/revision/%s?blobs=true' % (repo, sha)
    with opener(url, timeout=120) as src:
        info = json.load(src)
    if info.get('sha') != sha:
        raise ValueError('Hub returned a different revision')
    return info['siblings']


def build_manifest(repo, sha, opener=urllib.request.urlopen):
    """Complete manifest of an immutable revision: LFS SHA256 from the Hub, small files hashed here."""
    files = []
    for item in sorted(hub_files(repo, sha, opener), key=lambda x: x['rfilename']):
        name = item['rfilename']
        if name in MANIFEST_EXCLUDE:
            continue
        lfs = item.get('lfs') or {}
        if lfs.get('sha256'):
            files.append(dict(f=name, size=lfs['size'], sha256=lfs['sha256']))
            continue
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (repo, sha, urllib.parse.quote(name, safe='/'))
        with opener(url, timeout=120) as src:
            body = src.read()
        files.append(dict(f=name, size=len(body), sha256=hashlib.sha256(body).hexdigest()))
    manifest = dict(repo=repo, sha=sha, files=files, schema=1)
    validate_manifest(manifest)
    return manifest


def manifest_for(kind):
    if kind not in KINDS:
        raise ValueError('kind must be one of ' + ', '.join(KINDS))
    return json.loads((Path(__file__).resolve().parents[1] / 'manifests' / (kind + '.json')).read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'verify', 'manifest'))
    parser.add_argument('kind', help='target, abliterated or drafter; for manifest: <repo>@<revision>')
    parser.add_argument('directory', type=Path, help='checkpoint directory; for manifest: output JSON')
    opts = parser.parse_args()
    if opts.command == 'manifest':
        repo, _, rev = opts.kind.partition('@')
        built = build_manifest(repo, rev)
        opts.directory.write_text(json.dumps(built, indent=2) + '\n')
        opts.directory.with_suffix('.sha256').write_text(''.join('%s  %s\n' % (f['sha256'], f['f']) for f in built['files']))
    else:
        (download if opts.command == 'download' else verify)(opts.directory, manifest_for(opts.kind))
