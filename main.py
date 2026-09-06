import cv2
import os
import threading
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

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

def analyze_frame(frame):
    global faces, processing
    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender'],
            detector_backend='retinaface',   # was 'opencv' — this was the bug
            enforce_detection=False
        )
        if isinstance(results, dict):
            results = [results]
        with faces_lock:
            faces = results if results else []
        print(f"[analyze_frame] faces detected: {len(faces)}")
    except Exception as e:
        print(f"[analyze_frame] ERROR: {type(e).__name__}: {e}")
    finally:
        processing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    small_frame = cv2.resize(frame, (640, 480))

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

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
