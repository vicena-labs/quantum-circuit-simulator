"""Fail-closed static check of files selected for a public source tree.

This is an additional guard, not an exhaustive secret or copyright audit.
It reports only filenames and detector labels, never possible secret contents.
Run at the repository root or supply --root on the command line.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

# This list is about repository distribution, not about source code that
# names ordinary domain concepts like security, token budgets or secrets.
PRIVATE_FILENAMES = {
    'SOURCE_PROVENANCE.json', 'RELEASE_APPROVAL.json',
    'RELEASE_READINESS.md', 'SECURITY_AUDIT.md',
    'QUALIFICATION.json', 'BUNDLE_SHA256.json',
}
PRIVATE_COMPONENTS = {
    '.git', '.venv', '.pytest_cache', '__pycache__', 'build', 'dist',
}
BINARY_SUFFIXES = {'.pyc', '.pyo', '.so', '.pyd', '.dll', '.dylib',
                   '.a', '.o', '.obj', '.exe', '.pem', '.p12', '.pfx', '.key'}

# Standard published token formats. Assemble the private-key delimiter to
# avoid embedding a valid-looking key block in the scanner itself.
DETECTORS: dict[str, re.Pattern[str]] = {
    'private_key_header': re.compile('-----' + 'BEGIN\\s+(?:OPENSSH|RSA|EC|DSA|ENCRYPTED|PGP|PRIVATE)(?:\\s+[A-Z]+)*\\s+KEY-----'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b'),
    'github_credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{34,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'openai_or_anthropic_key': re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b'),
    'google_api_key': re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    'slack_token': re.compile(r'\bxox[bpars]-[A-Za-z0-9-]{12,}\b'),
    'stripe_live_key': re.compile(r'\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b'),
    'gitlab_token': re.compile(r'\bglpat-[A-Za-z0-9_-]{20,}\b'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b'),
    'authenticated_url': re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s@/:]{2,}:[^\s@/]{4,}@[^\s/]+'),
    'credential_assignment': re.compile(
        r'(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password|private[_-]?key)'
        r'\s*[:=]\s*[\'\"][^\'\"\s]{12,}[\'\"]'
    ),
    'absolute_build_path': re.compile(r'(?<![a-z])/(?:mnt/data|workspace|home/[A-Za-z0-9._-]+|Users/[A-Za-z0-9._-]+|root)/(?:[A-Za-z0-9._-]+/)?'),
    'private_ssh_url': re.compile(r'\bgit@[A-Za-z0-9.-]+:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'),
}

TEXT_SUFFIXES = {'.py', '.cpp', '.h', '.hpp', '.md', '.json', '.toml',
                 '.ini', '.yml', '.yaml', '.txt', '.cfg', '.in', '.sh', '.cff', '.jl', '.jsonl'}


def selected_files(root: Path, tracked_only: bool) -> list[Path]:
    """Select exactly tracked files in checkout mode; otherwise inspect the tree.

    Git metadata is not source and cannot be tracked. Strict directory mode
    still rejects it, so archives cannot accidentally carry private history.
    """
    if not tracked_only:
        return sorted(root.rglob('*'))
    top = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
        check=True, capture_output=True, text=True, timeout=15,
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError('tracked mode requires the repository root')
    listing = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '--cached', '-z'],
        check=True, capture_output=True, timeout=15,
    )
    paths = []
    for name in listing.stdout.decode('utf-8').split('\0'):
        if not name:
            continue
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe tracked path')
        paths.append(root / rel)
    if not paths:
        raise ValueError('no tracked source files to check')
    return sorted(set(paths))


def scan_tree(root: Path, *, tracked_only: bool = False) -> dict:
    findings: list[dict[str, str]] = []
    checked = 0
    root = root.resolve()
    try:
        files = selected_files(root, tracked_only)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        return {'ok': False, 'text_files_checked': 0,
                'findings': [{'file': '.', 'reason': 'source_selection_failed'}],
                'limitations': 'source selection failed; no absence-of-secrets claim'}
    for file in files:
        rel = file.relative_to(root)
        # A symlink to either a file or a directory may escape this tree.
        if file.is_symlink():
            findings.append({'file': rel.as_posix(), 'reason': 'symlink_not_reviewed'})
            continue
        if not file.is_file():
            if tracked_only:
                findings.append({'file': rel.as_posix(), 'reason': 'tracked_file_missing'})
            continue
        if tracked_only and (not file.resolve().is_relative_to(root) or
                             any(parent.is_symlink() for parent in file.parents
                                 if parent != root and parent.is_relative_to(root))):
            findings.append({'file': rel.as_posix(), 'reason': 'unsafe_tracked_file_path'})
            continue
        if file.name in PRIVATE_FILENAMES:
            findings.append({'file': rel.as_posix(), 'reason': 'private_release_record'})
        if any(part in PRIVATE_COMPONENTS or (part.startswith('.env') and part != '.env.example')
               for part in rel.parts):
            findings.append({'file': rel.as_posix(), 'reason': 'excluded_build_or_environment_path'})
            continue
        if file.suffix.lower() in BINARY_SUFFIXES or file.name.endswith('.egg-info'):
            findings.append({'file': rel.as_posix(), 'reason': 'binary_or_key_material'})
            continue
        # Bundled data is gzip-compressed JSON: scanned as its decompressed text.
        compressed = file.name.endswith('.json.gz')
        if (file.suffix.lower() not in TEXT_SUFFIXES and not compressed
                and file.name not in {'.gitignore', 'MANIFEST.in', 'LICENSE', 'NOTICE'}):
            findings.append({'file': rel.as_posix(), 'reason': 'unreviewed_file_type'})
            continue
        try:
            # Do not load unbounded fixtures into memory during release CI.
            if file.stat().st_size > 2_000_000:
                findings.append({'file': rel.as_posix(), 'reason': 'large_unreviewed_file'})
                continue
            if compressed:
                import gzip
                with gzip.open(file, 'rt', encoding='utf-8') as stream:
                    contents = stream.read(50_000_000)
                json.loads(contents)   # must be complete, well-formed JSON
            else:
                contents = file.read_text(encoding='utf-8')
        except (UnicodeError, OSError, ValueError):
            findings.append({'file': rel.as_posix(), 'reason': 'unreadable_text_file'})
            continue
        checked += 1
        for label, pattern in DETECTORS.items():
            if pattern.search(contents):
                findings.append({'file': rel.as_posix(), 'reason': label})
    return {'ok': len(findings) == 0, 'text_files_checked': checked,
            'findings': findings,
            'limitations': 'pattern scan only; cannot prove absence of unknown secrets or rights issues'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tracked-only', action='store_true',
                        help='Scan tracked source in a Git checkout; exclude Git metadata.')
    args = parser.parse_args()
    result = scan_tree(args.root.resolve(), tracked_only=args.tracked_only)
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
