import json
from pathlib import Path

from confusion_demo.models import TeachingCase


def load_case(relative_path: str) -> TeachingCase:
    project_root = Path(__file__).resolve().parent.parent
    data = json.loads((project_root / relative_path).read_text(encoding="utf-8"))
    return TeachingCase(**data)
