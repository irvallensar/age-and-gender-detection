from deepface import DeepFace

# Load models before starting detection (one-time load)
DeepFace.build_model("Facenet")
DeepFace.build_model("Age")
DeepFace.build_model("Gender")
