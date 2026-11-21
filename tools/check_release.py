"""Mechanical public-release guard. Does NOT determine source ownership."""
from pathlib import Path
import json
import re

root=Path(__file__).resolve().parents[1]
pyproject=(root/'pyproject.toml').read_text(encoding='utf-8')
match=re.search(r'^license\s*=\s*"([^"]+)"\s*$',pyproject,flags=re.MULTILINE)
license_expression=match.group(1) if match else None
issues=[]
if not (root/'LICENSE').is_file():
    issues.append('No approved LICENSE file is present.')
if not license_expression or 'Pending' in license_expression or license_expression.startswith('LicenseRef-'):
    issues.append('Package metadata does not declare an approved license.')
if (root/'LICENSE-PENDING.md').exists():
    issues.append('Pending-license marker must be resolved before publication.')
if not (root/'.github'/'SECURITY.md').exists():
    issues.append('Security disclosure guidance is missing.')
print(json.dumps({'release_metadata_qualified':not issues,
                  'manual_rights_and_name_review_still_required':True,
                  'issues':issues},indent=2))
raise SystemExit(1 if issues else 0)
