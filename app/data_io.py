from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_race_history(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load race history from a JSON Lines file (one JSON object per line).
    Returns [] if the file doesn't exist or contains malformed lines.
    """
    p = Path(file_path)
    if not p.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines, continue loading others
                continue
    return rows
