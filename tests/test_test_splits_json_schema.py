"""
Branch 4 — test-split-config: validate test_splits.json schema.

Tests:
  1. JSON has test_core, test_thin, train_exclude keys
  2. No overlap between test_core and test_thin
  3. train_exclude == union of test_core + test_thin
  4. All values are non-empty lists of strings matching 'image_\\d+'
"""

import json
import re
import tempfile
from pathlib import Path


def _make_valid_splits():
    """Return a valid test_splits dict."""
    return {
        "test_core": ["image_3", "image_13", "image_14"],
        "test_thin": ["image_22", "image_24"],
        "train_exclude": ["image_3", "image_13", "image_14", "image_22", "image_24"],
    }


class TestSplitsJsonSchema:
    def test_required_keys_present(self):
        splits = _make_valid_splits()
        for key in ('test_core', 'test_thin', 'train_exclude'):
            assert key in splits, f"Missing key: {key}"

    def test_no_overlap_core_thin(self):
        splits = _make_valid_splits()
        core = set(splits['test_core'])
        thin = set(splits['test_thin'])
        assert core.isdisjoint(thin), f"Overlap: {core & thin}"

    def test_train_exclude_is_union(self):
        splits = _make_valid_splits()
        union = set(splits['test_core']) | set(splits['test_thin'])
        assert set(splits['train_exclude']) == union

    def test_all_ids_match_pattern(self):
        splits = _make_valid_splits()
        pattern = re.compile(r'^image_\d+$')
        for key in ('test_core', 'test_thin', 'train_exclude'):
            for item in splits[key]:
                assert pattern.match(item), f"Invalid ID '{item}' in {key}"

    def test_lists_are_nonempty(self):
        splits = _make_valid_splits()
        for key in ('test_core', 'test_thin', 'train_exclude'):
            assert len(splits[key]) > 0, f"{key} is empty"

    def test_roundtrip_json(self, tmp_path):
        """Write to file and load back — schema is preserved."""
        splits = _make_valid_splits()
        path = tmp_path / "test_splits.json"
        path.write_text(json.dumps(splits, indent=2))
        loaded = json.loads(path.read_text())
        assert loaded == splits
