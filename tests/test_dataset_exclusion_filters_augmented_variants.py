"""
Branch 4 — test-split-config: verify exclude_list filters augmented variants.

Tests:
  1. Excluding 'image_1' removes image_1.png AND image_1_aug00.png, image_1_aug01.png
  2. Non-excluded images are kept
  3. Empty exclude list keeps all images
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_segmentation import WhiteboardDataset
from tests.helpers.synthetic_data import populate_synthetic_dataset


class TestDatasetExclusionFiltersAugments:
    def test_exclude_removes_originals_and_augments(self, tmp_path):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=5, n_augments=3,
                                   img_size=(32, 32), start_id=1)

        # Exclude image_1 and image_3
        ds = WhiteboardDataset(ds_root, train=True, augment=False,
                               img_size=(32, 32), exclude_list=["image_1", "image_3"])
        # image_1 + 3 augments + image_3 + 3 augments = 8 excluded
        # remaining: image_2, image_4, image_5 + their augments = 3*4 = 12
        # After 80/20 train split on 12 items → train ≈ 9
        for fname in ds.files:
            stem = Path(fname).stem
            assert not stem.startswith("image_1_") and stem != "image_1", \
                f"image_1 variant should be excluded: {fname}"
            assert not stem.startswith("image_3_") and stem != "image_3", \
                f"image_3 variant should be excluded: {fname}"

    def test_no_exclusion_keeps_all(self, tmp_path):
        ds_root = tmp_path / "dataset"
        info = populate_synthetic_dataset(ds_root, n_originals=3, n_augments=2,
                                          img_size=(32, 32), start_id=1)
        ds_all = WhiteboardDataset(ds_root, train=True, augment=False,
                                    img_size=(32, 32), exclude_list=None)
        ds_empty = WhiteboardDataset(ds_root, train=True, augment=False,
                                      img_size=(32, 32), exclude_list=[])
        assert len(ds_all.files) == len(ds_empty.files)

    def test_exclude_list_reduces_total_count(self, tmp_path):
        ds_root = tmp_path / "dataset"
        populate_synthetic_dataset(ds_root, n_originals=5, n_augments=2,
                                   img_size=(32, 32), start_id=1)
        ds_full = WhiteboardDataset(ds_root, train=True, augment=False,
                                     img_size=(32, 32), exclude_list=None)
        ds_excl = WhiteboardDataset(ds_root, train=True, augment=False,
                                     img_size=(32, 32), exclude_list=["image_2"])
        # image_2 + 2 augments = 3 files excluded before split
        assert len(ds_excl.files) < len(ds_full.files)
