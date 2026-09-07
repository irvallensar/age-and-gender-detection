import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # must be set before importing tensorflow — fixes KerasTensor error on TF 2.16+

import re
import gc
import json
import random
import argparse
import tensorflow as tf
from deepface import DeepFace
from tqdm import tqdm

DATASET_DIR = "utkface_data"          # folder of UTKFace images
CLEAR_SESSION_EVERY = 500              # frees TF/Keras graph state periodically


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"processed": [], "age_errors": [], "gender_correct": 0, "gender_total": 0}


def save_checkpoint(state, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        json.dump(state, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate this many images (for a quick test run before committing to the full dataset)")
    parser.add_argument("--detector", type=str, default="skip",
                         help="DeepFace detector_backend to use, e.g. 'skip', 'retinaface', 'mtcnn', 'ssd'")
    args = parser.parse_args()

    # separate checkpoint per detector backend — prevents mixing results from different configs
    checkpoint_file = f"eval_checkpoint_{args.detector}.json"

    state = load_checkpoint(checkpoint_file)
    processed_set = set(state["processed"])

    all_files = sorted(os.listdir(DATASET_DIR))
    if args.limit:
        random.seed(42)  # fixed seed — reproducible sample, but no longer biased toward "100_..." filenames
        random.shuffle(all_files)
        all_files = all_files[:args.limit]

    remaining = [f for f in all_files if f not in processed_set]

    if not remaining:
        print("Nothing left to process — all files already in checkpoint.")
    else:
        print(f"{len(processed_set)} already done, {len(remaining)} remaining.")

    for i, filename in enumerate(tqdm(remaining, desc="Evaluating")):
        match = re.match(r"(\d+)_(\d+)_", filename)
        if not match:
            state["processed"].append(filename)
            continue

        true_age = int(match.group(1))
        true_gender = "Man" if match.group(2) == "0" else "Woman"  # UTKFace: 0=male, 1=female
        filepath = os.path.join(DATASET_DIR, filename)

        try:
            result = DeepFace.analyze(
                filepath,
                actions=['age', 'gender'],
                detector_backend=args.detector,
                enforce_detection=False
            )
            if isinstance(result, list):
                result = result[0]

            predicted_age = result['age']
            predicted_gender = result['dominant_gender']

            state["age_errors"].append(abs(predicted_age - true_age))
            state["gender_total"] += 1
            if predicted_gender == true_gender:
                state["gender_correct"] += 1

            if i < 10:  # print first 10 for a quick sanity check
                tqdm.write(f"{filename}: true_age={true_age}, pred_age={predicted_age}, "
                           f"true_gender={true_gender}, pred_gender={predicted_gender}")

        except Exception as e:
            tqdm.write(f"Skipped {filename}: {e}")

        state["processed"].append(filename)

        # Periodic cleanup — prevents TF/Keras graph state from accumulating across
        # thousands of iterations, which is what was driving swap usage up.
        if (i + 1) % CLEAR_SESSION_EVERY == 0:
            tf.keras.backend.clear_session()
            gc.collect()
            save_checkpoint(state, checkpoint_file)  # checkpoint so a crash/interrupt doesn't lose progress

    save_checkpoint(state, checkpoint_file)

    if state["age_errors"]:
        mae = sum(state["age_errors"]) / len(state["age_errors"])
        print(f"\nAge MAE: {mae:.2f} years over {len(state['age_errors'])} images")
    else:
        print("\nNo age results collected.")

    if state["gender_total"]:
        gender_accuracy = (state["gender_correct"] / state["gender_total"]) * 100
        print(f"Gender accuracy: {gender_accuracy:.2f}% over {state['gender_total']} images")
    else:
        print("No gender results collected.")


if __name__ == "__main__":
    main()
