"""
calibrate_age.py — fits a linear correction to reduce systematic age bias.

Usage: run evaluate_accuracy.py first (with --limit ~3000, --detector retinaface)
to produce a checkpoint file with raw (true_age, pred_age) pairs, then run this.
"""
import json
import random

CHECKPOINT_FILE = "eval_checkpoint_retinaface.json"

with open(CHECKPOINT_FILE, "r") as f:
    state = json.load(f)

pairs = state["pairs"]
random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.7)
train_pairs = pairs[:split]
test_pairs = pairs[split:]

# Fit a simple linear correction: true_age ≈ a * pred_age + b
n = len(train_pairs)
sum_x = sum(p["pred_age"] for p in train_pairs)
sum_y = sum(p["true_age"] for p in train_pairs)
sum_xy = sum(p["pred_age"] * p["true_age"] for p in train_pairs)
sum_x2 = sum(p["pred_age"] ** 2 for p in train_pairs)

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - a * sum_x) / n

print(f"Fitted correction: corrected_age = {a:.3f} * pred_age + {b:.3f}")

# Evaluate on held-out test set
raw_mae = sum(abs(p["pred_age"] - p["true_age"]) for p in test_pairs) / len(test_pairs)
corrected_mae = sum(abs((a * p["pred_age"] + b) - p["true_age"]) for p in test_pairs) / len(test_pairs)

print(f"\nOn {len(test_pairs)} held-out test images:")
print(f"  Raw MAE:       {raw_mae:.2f} years")
print(f"  Corrected MAE: {corrected_mae:.2f} years")
print(f"  Improvement:   {(raw_mae - corrected_mae) / raw_mae * 100:.1f}%")
