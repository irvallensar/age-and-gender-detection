# Age and Gender Detection

Real-time age and gender detection from a webcam feed, built with [DeepFace](https://github.com/serengil/deepface) and OpenCV. Face detection and analysis run on a background thread so the video feed stays responsive.

## Features

- Live webcam face detection using RetinaFace
- Real-time age and gender prediction per detected face
- Multithreaded analysis (analysis runs off the main video loop to avoid blocking the feed)
- Standalone accuracy evaluation script against a labeled dataset (UTKFace)

## Video Demo

[▶️ Watch the demo video](https://docs.google.com/file/d/1rXSH8MmUygn3Q2WhNYrfuaU3UNUWpsI-/view)

## Requirements

- Python 3.8+
- A working webcam
- ~1–2 GB free disk space (model weights + dataset, if running evaluation)

## Installation

```bash
pip install deepface opencv-python datasets
```

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Live webcam app — detects faces and predicts age/gender in real time |
| `download_models.py` | Preloads DeepFace's Age/Gender/Facenet models before running `main.py` |
| `download_dataset.py` | One-time script to download the UTKFace dataset (via Hugging Face) for evaluation |
| `evaluate_accuracy.py` | Runs the model against UTKFace and reports age MAE and gender accuracy |

## Usage

### 1. Preload the models (optional, but avoids a delay on first run)

```bash
python download_models.py
```

### 2. Run the live detector

```bash
python main.py
```

- A window opens showing your webcam feed with a bounding box, predicted age, and predicted gender drawn over any detected face.
- Press `q` to quit.

### 3. (Optional) Evaluate accuracy on a labeled dataset

This project's own model is **not trained** — it uses DeepFace's pretrained Age/Gender models. To get real, measured accuracy numbers (rather than the model authors' published benchmarks), run the model against a labeled dataset:

```bash
python download_dataset.py
python evaluate_accuracy.py
```

`download_dataset.py` pulls the UTKFace dataset (age/gender-labeled face images) via Hugging Face. `evaluate_accuracy.py` runs DeepFace on every image and reports:

- **Age MAE** (mean absolute error, in years)
- **Gender accuracy** (%)
- Number of images evaluated

## Notes on Accuracy

This project uses DeepFace's pretrained Age and Gender models (VGG16-based, DEX architecture), not a custom-trained model. Two ways to report accuracy:

- **Published benchmark** (not measured by this project): ~97.44% gender accuracy, ±4.65 years age MAE, reported by the model's authors on the IMDB-WIKI dataset (500K images).
- **Measured by this project**: run `evaluate_accuracy.py` and report the numbers it prints, along with the dataset size used.

## Known Limitations

- Accuracy depends heavily on lighting, face angle, and camera quality — webcam conditions are typically worse than the curated datasets used for the published benchmarks above.
- RetinaFace (the detector used here) is more accurate than OpenCV's Haar cascade but slower — expect lower FPS on limited hardware.
- Age prediction is inherently approximate; a few years of error per prediction is expected even from well-performing models.

## Credits

- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil
- [UTKFace dataset](https://susanqq.github.io/UTKFace/)
