"""
augment_data.py  —  Data Augmentation for Sign Language Training Data
======================================================================
Solves THREE problems:
  1. LIGHTING VARIATION  : Noise augmentations simulate what MediaPipe
                           outputs in low-light  (jittery, uncertain
                           landmarks) so the model learns to handle both
                           home (dim) and college (bright) conditions.
  2. MORE SAMPLES        : 14 augmentations per original sample → ×15
                           dataset size for your research paper.
  3. SIMILAR-SIGN CONFUSION: Rotation / scale variants teach the model
                           that the SHAPE matters, not position, making
                           it harder to confuse look-alike signs.

Usage:
  python augment_data.py               # augment all signs
  python augment_data.py --preview     # show stats without saving
"""

import numpy as np
import pickle
import os
import argparse

# ── Signs must match train_model.py ────────────────────────────────────────
SIGNS = [
    'my', 'name', 'is', 'ashish', 'hello', 'me', 'fine',
    'drink', 'eat', 'good', 'love', 'morning', 'please', 'thanks', 'you'
]
DATA_DIR = 'training_data'


# ───────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ───────────────────────────────────────────────────────────────────────────

def _reshape(flat):
    """63-dim flat vector → (21, 3) landmark array."""
    return flat.reshape(21, 3)


def _normalize(lm_21x3):
    """
    Re-apply the same normalization used during recording:
      • Translate so wrist (landmark 0) is at origin
      • Scale so max absolute value == 1
    Returns a 63-dim flat vector.
    """
    lm = lm_21x3.copy().astype(np.float64)
    lm -= lm[0]                          # wrist to origin
    max_val = np.max(np.abs(lm))
    if max_val > 1e-9:
        lm /= max_val
    return lm.flatten()


# ───────────────────────────────────────────────────────────────────────────
#  14 Augmentation techniques
# ───────────────────────────────────────────────────────────────────────────

def augment_sample(sample_63, rng=None):
    """
    Given one normalized 63-dim landmark vector, produce 14 augmented copies.

    Augmentation map:
    ┌───┬───────────────────────────────────────────────────────────────────┐
    │ # │ Technique                │ What it solves                        │
    ├───┼───────────────────────────────────────────────────────────────────┤
    │1-3│ Gaussian noise σ=0.008  │ Lighting jitter (normal conditions)   │
    │4-5│ Gaussian noise σ=0.018  │ Low-light uncertain detection         │
    │6-9│ 2-D rotation ±5°/±10°  │ Slight wrist tilt variation            │
    │10 │ Scale × 0.88            │ Hand farther from camera              │
    │11 │ Scale × 1.12            │ Hand closer to camera                 │
    │12 │ Horizontal mirror       │ Left/right-hand ambiguity             │
    │13 │ Z-axis noise σ=0.02    │ Depth / perspective variation         │
    │14 │ Random wrist tilt ±8°  │ Orientation robustness                │
    └───┴───────────────────────────────────────────────────────────────────┘
    """
    if rng is None:
        rng = np.random.default_rng()

    lm = _reshape(sample_63)
    augmented = []

    # ── 1-3  Mild Gaussian noise (lighting jitter) ─────────────────────────
    for _ in range(3):
        noisy = lm + rng.normal(0, 0.008, lm.shape)
        augmented.append(_normalize(noisy))

    # ── 4-5  Medium Gaussian noise (low-light uncertainty) ─────────────────
    for _ in range(2):
        noisy = lm + rng.normal(0, 0.018, lm.shape)
        augmented.append(_normalize(noisy))

    # ── 6-9  2-D rotation in XY plane around wrist (wrist = origin) ────────
    for angle_deg in [-10, -5, 5, 10]:
        a = np.radians(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        rot = np.array([[ca, -sa, 0],
                        [sa,  ca, 0],
                        [0,   0,  1]])
        rotated = (rot @ lm.T).T
        augmented.append(_normalize(rotated))

    # ── 10-11  Scale variation ──────────────────────────────────────────────
    for scale in [0.88, 1.12]:
        augmented.append(_normalize(lm * scale))

    # ── 12  Horizontal mirror (flip x) ─────────────────────────────────────
    mirrored = lm.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    augmented.append(_normalize(mirrored))

    # ── 13  Z-axis noise (depth variation) ─────────────────────────────────
    z_noisy = lm.copy()
    z_noisy[:, 2] += rng.normal(0, 0.02, 21)
    augmented.append(_normalize(z_noisy))

    # ── 14  Random wrist-tilt rotation around Y-axis ───────────────────────
    a = np.radians(rng.uniform(-8, 8))
    ca, sa = np.cos(a), np.sin(a)
    rot_y = np.array([[ca,  0, sa],
                      [0,   1,  0],
                      [-sa, 0, ca]])
    tilted = (rot_y @ lm.T).T
    augmented.append(_normalize(tilted))

    return augmented   # exactly 14 copies


# ───────────────────────────────────────────────────────────────────────────
#  Main augmentation pipeline
# ───────────────────────────────────────────────────────────────────────────

def augment_all(preview=False, seed=42):
    """
    For every sign that has a <sign>.pkl file:
      - Load original samples
      - Generate 14 augmentations per sample
      - Save combined data to <sign>_augmented.pkl

    Parameters
    ----------
    preview : bool  – if True, only print stats, do NOT save
    seed    : int   – random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    header = "PREVIEW MODE  (no files saved)" if preview else "SAVING augmented files"

    print("\n" + "=" * 72)
    print("   DATA AUGMENTATION  —  Sign Language Recognition")
    print("=" * 72)
    print(f"   Mode : {header}")
    print(f"   Seed : {seed}  (results are reproducible)")
    print()
    print("   Augmentation techniques:")
    print("     1-3.  Gaussian noise σ=0.008 (×3) → lighting jitter")
    print("     4-5.  Gaussian noise σ=0.018 (×2) → low-light uncertainty")
    print("     6-9.  2-D rotation ±5°/±10°  (×4) → wrist-tilt robustness")
    print("     10-11.Scale ×0.88 / ×1.12   (×2) → distance variation")
    print("     12.   Horizontal mirror       (×1) → hand-side ambiguity")
    print("     13.   Z-axis noise σ=0.02    (×1) → depth variation")
    print("     14.   Random Y-tilt ±8°      (×1) → orientation robustness")
    print()
    print(f"   Each sample  →  1 original + 14 augmented  =  ×15 dataset")
    print("=" * 72)
    print(f"\n   {'Sign':<12} {'Original':>10} {'+ Augmented':>12} {'Total':>8}")
    print(f"   {'-'*46}")

    total_orig = 0
    total_aug  = 0

    for sign in SIGNS:
        orig_file = os.path.join(DATA_DIR, f'{sign}.pkl')

        if not os.path.exists(orig_file):
            print(f"   {sign:<12} {'(no file)':>32}")
            continue

        with open(orig_file, 'rb') as f:
            original_data = pickle.load(f)

        if len(original_data) == 0:
            print(f"   {sign:<12} {'(empty)':>32}")
            continue

        # Generate augmented samples
        aug_samples = []
        for sample in original_data:
            aug_samples.extend(augment_sample(np.array(sample), rng=rng))

        orig_n = len(original_data)
        aug_n  = len(aug_samples)
        total  = orig_n + aug_n

        print(f"   {sign:<12} {orig_n:>10} {aug_n:>12} {total:>8}")
        total_orig += orig_n
        total_aug  += aug_n

        if not preview:
            combined      = list(original_data) + aug_samples
            out_file      = os.path.join(DATA_DIR, f'{sign}_augmented.pkl')
            with open(out_file, 'wb') as f:
                pickle.dump(combined, f)

    grand_total = total_orig + total_aug
    factor      = grand_total / total_orig if total_orig > 0 else 0

    print(f"   {'-'*46}")
    print(f"   {'TOTAL':<12} {total_orig:>10} {total_aug:>12} {grand_total:>8}")
    print()
    print(f"   Expansion factor : {factor:.1f}x")

    if not preview:
        print()
        print("   Saved files:")
        for sign in SIGNS:
            aug_file = os.path.join(DATA_DIR, f'{sign}_augmented.pkl')
            if os.path.exists(aug_file):
                size_kb = os.path.getsize(aug_file) / 1024
                print(f"     training_data/{sign}_augmented.pkl  ({size_kb:.1f} KB)")

    print()
    print("   NEXT STEP:")
    print("     python train_model.py --augmented")
    print("     (trains using the _augmented.pkl files)")
    print("=" * 72)

    return total_orig, grand_total


# ───────────────────────────────────────────────────────────────────────────
#  Research-paper summary
# ───────────────────────────────────────────────────────────────────────────

def print_paper_summary(orig, total):
    print("\n" + "=" * 72)
    print("   RESEARCH PAPER — Dataset Statistics")
    print("=" * 72)
    print(f"   Signs in vocabulary  : {len(SIGNS)}")
    print(f"   Original samples     : {orig}")
    print(f"   After augmentation   : {total}")
    print(f"   Augmentation factor  : {total / orig:.1f}x")
    print()
    print("   Technique breakdown (per original sample):")
    print("   ┌─────────────────────────────────────┬────────┐")
    print("   │ Technique                           │ Count  │")
    print("   ├─────────────────────────────────────┼────────┤")
    print("   │ Gaussian noise (σ=0.008)            │   ×3   │")
    print("   │ Gaussian noise (σ=0.018, low-light) │   ×2   │")
    print("   │ 2-D rotation (±5°, ±10°)           │   ×4   │")
    print("   │ Scale variation (×0.88, ×1.12)     │   ×2   │")
    print("   │ Horizontal mirror                   │   ×1   │")
    print("   │ Z-axis (depth) noise               │   ×1   │")
    print("   │ Wrist-tilt rotation (±8°)          │   ×1   │")
    print("   ├─────────────────────────────────────┼────────┤")
    print("   │ Total augmentations per sample      │  ×14   │")
    print("   └─────────────────────────────────────┴────────┘")
    print("=" * 72)


# ───────────────────────────────────────────────────────────────────────────
#  Entry point
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment sign language training data')
    parser.add_argument('--preview', action='store_true',
                        help='Show stats without saving files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    orig_count, total_count = augment_all(preview=args.preview, seed=args.seed)

    if not args.preview and orig_count > 0:
        print_paper_summary(orig_count, total_count)
