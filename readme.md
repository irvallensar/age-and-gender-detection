# Age and Gender Detection 

The project is a real-time Age and Gender Detection system developed in Python using OpenCV, DeepFace, TensorFlow/Keras, and pre-trained deep learning models. It uses RetinaFace to detect faces and extract bounding boxes, while DeepFace’s CNN-based models estimate the age and gender of each detected face. The system supports multiple face detection, dynamic labels, real-time webcam processing, and performs reasonably well even in relatively low-light conditions, with age predictions reported to be accurate within approximately ±5 years. To improve real-time performance, the program uses multithreading and frame skipping to reduce webcam lag. However, its predictions can be affected by lighting, facial angles, expressions, training-data bias, and limitations of the pre-trained models, while gender classification may be less reliable for ambiguous facial features and face tracking can experience delays due to DeepFace inference time.


**Video Demo**

