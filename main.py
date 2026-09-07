import cv2
import os
import time
import argparse
import threading
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--sync", action="store_true",
                     help="Run analysis synchronously (blocking) in the main loop, instead of on a background thread. Use this to measure the non-threaded baseline FPS.")
args = parser.parse_args()

weights_path = os.path.expanduser("~/.deepface/weights/")
os.makedirs(weights_path, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

faces = []
faces_lock = threading.Lock()
processing = False
frame_skip = 2
frame_count = 0

executor = ThreadPoolExecutor(max_workers=1)

# FPS tracking
prev_time = time.time()
fps = 0.0
fps_samples = []          # collected for the run summary
BENCHMARK_FRAMES = 300    # print an average after this many frames, then keep running


def analyze_frame(frame):
    global faces, processing
    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        if isinstance(results, dict):
            results = [results]
        with faces_lock:
            faces = results if results else []
    except Exception as e:
        print(f"[analyze_frame] ERROR: {type(e).__name__}: {e}")
    finally:
        processing = False


mode_label = "SYNC (no threading)" if args.sync else "THREADED"
print(f"Running in {mode_label} mode. Benchmarking first {BENCHMARK_FRAMES} processed frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    small_frame = cv2.resize(frame, (640, 480))

    if args.sync:
        # Blocking call — the loop cannot draw/display until analysis finishes.
        analyze_frame(small_frame.copy())
    else:
        if not processing:
            processing = True
            executor.submit(analyze_frame, small_frame.copy())

    with faces_lock:
        current_faces = list(faces)

    if current_faces:
        for face in current_faces:
            region = face.get('region', {})
            x, y, w, h = int(region.get('x', 0)), int(region.get('y', 0)), int(region.get('w', 0)), int(region.get('h', 0))
            if w == 0 or h == 0:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            age = face.get('age', 'N/A')
            gender = face.get('dominant_gender', 'N/A')
            cv2.putText(frame, f"{gender}, Age: {age}", (x, max(y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    fps_samples.append(fps)

    cv2.putText(frame, f"FPS: {fps:.1f} ({mode_label})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if len(fps_samples) == BENCHMARK_FRAMES:
        avg_fps = sum(fps_samples) / len(fps_samples)
        print(f"\n--- Benchmark result ({mode_label}) ---")
        print(f"Average FPS over {BENCHMARK_FRAMES} frames: {avg_fps:.2f}")
        print("Continuing to run — press 'q' to quit.\n")

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
