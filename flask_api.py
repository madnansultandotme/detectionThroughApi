from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tensorflow as tf

app = Flask(__name__)
#Test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'}), 200

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Directory to store face encodings
ENCODINGS_DIR = "face_encodings"
if not os.path.exists(ENCODINGS_DIR):
    os.makedirs(ENCODINGS_DIR)

# Path to your local FaceNet model file
model_file = r"C:\Users\adnan\OneDrive\Desktop\NCP\detectionThroughApi\facenet\facenet.pb"

# Load the FaceNet model from .pb file
def load_graph(model_file):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

    return graph

# Load the graph
graph = load_graph(model_file)

# Start a session
sess = tf.compat.v1.Session(graph=graph)

# Define input and output tensor names
input_tensor_name = 'input:0'  # Adjust this name according to your model's input
output_tensor_name = 'embeddings:0'  # Adjust this name according to your model's output
phase_train_tensor_name = 'phase_train:0'  # Phase train tensor

def process_frame(frame):
    """Process a video frame to detect faces."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    detections = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            detections.append(bbox)

    return detections

def encode_face(frame):
    """Encode a face using FaceNet."""
    # Preprocess the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb_frame, (160, 160))
    face = np.expand_dims(face, axis=0)
    face = (face - 127.5) / 128.0

    # Run the model to get embeddings
    embeddings = sess.run(output_tensor_name, feed_dict={
        input_tensor_name: face,
        phase_train_tensor_name: False
    })
    return embeddings.flatten()

@app.route('/add_user', methods=['POST'])
def add_user():
    # Get the user name from the request
    user_name = request.form['name']
    files = request.files.getlist('file')

    encodings = []
    for file in files:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Encode the face
        encoding = encode_face(frame)
        if encoding is not None:
            encodings.append(encoding)

    if encodings:
        # Average the encodings
        average_encoding = np.mean(encodings, axis=0)

        # Save the average encoding with the user name
        with open(f"{ENCODINGS_DIR}/{user_name}.pkl", 'wb') as f:
            pickle.dump(average_encoding, f)
        return jsonify({'message': f'User {user_name} added successfully!'}), 200
    else:
        return jsonify({'message': 'No face detected in the images.'}), 400

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['file'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Encode the face in the frame
    encoding = encode_face(frame)

    if encoding is None:
        return jsonify({'message': 'No face detected.'}), 400

    # Load known faces and compare
    known_encodings = []
    known_names = []

    print("Loading known encodings...")
    for filename in os.listdir(ENCODINGS_DIR):
        with open(f"{ENCODINGS_DIR}/{filename}", 'rb') as f:
            known_encodings.append(pickle.load(f))
            known_names.append(filename.split('.')[0])
            print(f"Loaded encoding for {filename}")

    # Calculate distances between the encoding and known encodings
    distances = np.linalg.norm(known_encodings - encoding, axis=1)
    print(f"Distances: {distances}")
    min_distance = np.min(distances)
    threshold = 0.7  # Set an appropriate threshold for your model
    name = "Unknown"

    if min_distance < threshold:
        min_index = np.argmin(distances)
        name = known_names[min_index]

    print(f"Recognized: {name} with distance {min_distance}")
    return jsonify({'name': name}), 200

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',port=5000,
        debug=True)
