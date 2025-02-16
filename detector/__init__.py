import cv2
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import threading

import logging

# Simple logging configuration
logging.basicConfig(
    filename='./app.log',  # Log file where messages will be saved
    level=logging.INFO,  # You can set it to DEBUG, ERROR, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'  # Simple format
)


# Model path
MODEL_PATH = 'models/df-model-2.tflite'

# Load model once at module level
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Lock for thread safety
interpreter_lock = threading.Lock()
# Load labels from JSON
labels = pd.read_json('labels/nl-new.json')
arr_labels = labels.to_numpy()

COLORS = np.random.randint(0, 255, size=(len(arr_labels), 3), dtype=np.uint8)

logging.info("init file loaded")

def preprocess_image(image_path, input_size):
    """Preprocess the input image for the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    resized_img = tf.image.resize(img, input_size)
    return resized_img[tf.newaxis, :], img


def set_input_tensor(image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(tensor_index)()[0][:] = image


def get_output_tensor(index):
    """Return the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details['index']))


def detect_objects(image, threshold):
    """Run object detection and return results"""
    with interpreter_lock:
        set_input_tensor(image)
        interpreter.invoke()

        scores = get_output_tensor(0)
        boxes = get_output_tensor(1)
        count = int(get_output_tensor(2))
        classes = get_output_tensor(3)

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            results.append({
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            })
    return results


def run_odt_and_draw_results(image_path, threshold=0.5):
    """Run object detection and visualize results"""
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    preprocessed_image, original_image = preprocess_image(image_path, (input_height, input_width))
    
    results = detect_objects(preprocessed_image, threshold)
    original_image_np = original_image.numpy().astype(np.uint8)

    problems, problem_colors, arr_labels_ids, arr_labels_extras = [], [], [], []
    
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin, xmax = int(xmin * original_image_np.shape[1]), int(xmax * original_image_np.shape[1])
        ymin, ymax = int(ymin * original_image_np.shape[0]), int(ymax * original_image_np.shape[0])
        class_id = obj['class_id']

        current_color = arr_labels[class_id-1][1]
        current_extra = arr_labels[class_id-1][2]

        if current_color == 'None':
            continue

        rgb_current_color = tuple(int(current_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), rgb_current_color, 3)

        label = f"{arr_labels[class_id-1][0]}: {obj['score'] * 100:.0f}%"
        cv2.putText(original_image_np, str(class_id-1), (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_current_color, 2)

        if arr_labels[class_id-1][0] not in problems:
            problems.append(arr_labels[class_id-1][0])
            problem_colors.append('#%02x%02x%02x' % rgb_current_color)
            arr_labels_ids.append(class_id-1)
            arr_labels_extras.append(current_extra)

    return original_image_np, problems, problem_colors, arr_labels_ids, arr_labels_extras


def detect(image_path):
    """Main function to run object detection"""
    DETECTION_THRESHOLD = 0.25
    detection_result_image = run_odt_and_draw_results(image_path, threshold=DETECTION_THRESHOLD)

    output_image_name = f'static/images/output/result-{uuid.uuid4().hex}.jpg'
    Image.fromarray(detection_result_image[0]).save(output_image_name)

    return json.dumps({
        'image': output_image_name,
        'problems': detection_result_image[1],
        'colors': detection_result_image[2],
        'ids': detection_result_image[3],
        'extras': detection_result_image[4]
    })
