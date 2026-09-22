import json
from pathlib import Path

CORPUS_DIR = Path("data/corpus")


def test_every_manifest_source_exists():
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    missing = [
        item["source_path"] for item in manifest if not (CORPUS_DIR / item["source_path"]).is_file()
    ]
    assert missing == []


def test_spain_scenario_is_consistent_and_old_markets_are_absent():
    paths = [CORPUS_DIR / "manifest.json", Path("data/eval_questions.json")]
    paths.extend(path for path in CORPUS_DIR.rglob("*") if path.is_file())
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "Spain" in combined
    assert "Chicago" not in combined
    assert "Italy" not in combined
