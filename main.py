import cv2
import os
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

weights_path = os.path.expanduser("~/.deepface/weights/")
os.makedirs(weights_path, exist_ok=True)

deepface_cache = os.path.join(weights_path, "retinaface.h5")
if not os.path.exists(deepface_cache):
    print("Downloading RetinaFace model...")
else:
    print("RetinaFace model already exists. Skipping download.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

age, gender = None, None
faces = []
processing = False
frame_skip = 2
frame_count = 0

executor = ThreadPoolExecutor(max_workers=1)


def analyze_frame(frame):
    """ Runs face analysis to detect age and gender """
    global faces, processing
    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender'],
            detector_backend='opencv',
            enforce_detection=False
        )

        if results and isinstance(results, list):
            faces = results  # Store multiple face results
        elif results and isinstance(results, dict):
            faces = [results]  # Handle single face result

    except Exception as e:
        print("Error during analysis:", str(e))
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
        executor.submit(analyze_frame, small_frame)

    if faces:
        for face in faces:
            region = face.get('region', {})

            if region:
                x, y, w, h = (
                    int(region.get('x', 0)),
                    int(region.get('y', 0)),
                    int(region.get('w', 0)),
                    int(region.get('h', 0))
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                age = face.get('age', 'N/A')
                gender = face.get('dominant_gender', 'N/A')

                cv2.putText(
                    frame,
                    f"{gender}, Age: {age}",
                    (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            else:
                print("No region detected for face.")

    else:
        print("No faces detected.")

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
