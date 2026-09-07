# Age and Gender Detection

Real-time age and gender detection from a webcam feed, built with [DeepFace](https://github.com/serengil/deepface) and OpenCV. Face detection and analysis run on a background thread so the video feed stays responsive.

## Features

- Live webcam face detection using RetinaFace
- Real-time age and gender prediction per detected face
- Multithreaded analysis (analysis runs off the main video loop to avoid blocking the feed), with a `--sync` flag to benchmark the non-threaded baseline
- Standalone accuracy evaluation script against a labeled dataset (UTKFace), with resumable checkpointing
- Post-hoc calibration script that corrects a systematic age-prediction bias

## Video Demo

[▶️ Watch the demo video](https://docs.google.com/file/d/1rXSH8MmUygn3Q2WhNYrfuaU3UNUWpsI-/view)

## Requirements

- Python 3.8+
- A working webcam
- ~1–2 GB free disk space (model weights + dataset, if running evaluation)

## Installation

```bash
pip install deepface opencv-python datasets tqdm tf-keras
```

`tf-keras` is required on TensorFlow 2.16+, which defaults to Keras 3 — DeepFace's internal model code is incompatible with Keras 3 and will fail with a `KerasTensor` error without it.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Live webcam app — detects faces and predicts age/gender in real time. Supports `--sync` for FPS benchmarking. |
| `dowload_models.py` | Preloads DeepFace's Age/Gender/Facenet models before running `main.py` |
| `download_dataset.py` | One-time script to download the UTKFace dataset (via Hugging Face) for evaluation |
| `evaluate_accuracy.py` | Runs the model against UTKFace and reports age MAE and gender accuracy. Supports `--limit` and `--detector` flags, and resumes from checkpoint if interrupted. |
| `calibrate_age.py` | Fits and validates a post-hoc linear correction for the age model's systematic bias |

## Usage

### 1. Preload the models (optional, but avoids a delay on first run)

```bash
python dowload_models.py
```

### 2. Run the live detector

```bash
python main.py
```

- A window opens showing your webcam feed with a bounding box, predicted age, and predicted gender drawn over any detected face.
- Press `q` to quit.

To benchmark the FPS impact of multithreading:

```bash
python main.py --sync    # synchronous baseline (no threading)
python main.py           # threaded (default)
```

Each prints an average FPS to the terminal after 300 processed frames.

### 3. Evaluate accuracy on a labeled dataset

```bash
python download_dataset.py
python evaluate_accuracy.py --detector retinaface
```

`download_dataset.py` pulls the UTKFace dataset (age/gender-labeled face images) via Hugging Face. `evaluate_accuracy.py` runs DeepFace on every image and reports age MAE and gender accuracy.

Useful flags:
- `--limit N` — evaluate only N randomly sampled images (for a quick test before committing to the full dataset)
- `--detector NAME` — choose the DeepFace detector backend (`retinaface` performed best in testing; `skip` is faster but less accurate on this dataset)

The script checkpoints progress every 500 images to `eval_checkpoint_<detector>.json`. If interrupted (crash, closed terminal, machine sleep), rerunning the same command resumes from where it left off instead of starting over. For long runs, use:

```bash
nohup caffeinate -i python3 evaluate_accuracy.py --detector retinaface > eval_output.log 2>&1 &
```

### 4. Correct systematic age bias (optional)

The pretrained age model tends to compress predictions toward the mean — it noticeably underestimates ages at the older end of the range. `calibrate_age.py` fits a linear correction on 70% of the evaluated data and validates it on the remaining 30%:

```bash
python calibrate_age.py
```

## Results

Measured against 23,707 images from the UTKFace dataset, using the RetinaFace detector backend:

| Metric | Raw | After calibration |
|---|---|---|
| Age MAE | 10.66 years | 9.16 years (−14.4%, validated on a held-out test set) |
| Gender accuracy | 83.58% | — (calibration only applies to age) |

For reference, DeepFace's own pretrained Age/Gender models report ~4.65 years MAE and ~97.44% gender accuracy on IMDB-WIKI (their original training/benchmark dataset, 500K images) — the gap reflects evaluating on a different dataset (UTKFace) under real-world webcam-adjacent conditions rather than a curated benchmark split.

### Live Inference Performance

Benchmarked on a Mac Studio (M3 Ultra), averaged over 300 processed frames:

| Mode | Display FPS | Avg. analysis time |
|---|---|---|
| Synchronous (no threading) | 0.53 | 3735 ms |
| Multithreaded (default) | 29.28 | 3612 ms |

Decoupling DeepFace analysis from the display loop via a background thread improved live display FPS by ~55×. Note the per-frame analysis cost itself is essentially unchanged between modes (~3.6–3.7 sec) — the improvement comes entirely from no longer blocking frame rendering while waiting on that analysis. Results will vary by hardware; run `python main.py --sync` and `python main.py` yourself to benchmark on your own machine.

## Known Limitations

- Accuracy depends heavily on lighting, face angle, and camera quality — webcam conditions are typically worse than the curated datasets used for the published benchmarks above.
- The age model exhibits regression-to-the-mean bias, particularly underestimating ages at the extreme older end of the range; `calibrate_age.py` partially corrects this but the correction is linear and may not fully capture non-linear bias patterns.
- Gender accuracy is notably lower for elderly subjects in testing — no correction has been applied for this.
- Age prediction is inherently approximate; a few years of error per prediction is expected even from well-performing models.

## Credits

- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil
- [UTKFace dataset](https://susanqq.github.io/UTKFace/)
