
# from xml.parsers.expat import model
import cv2
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd


model_path ='models/df-model.tflite'
# model_path ='models/detect.tflite'

# Load the labels into a list
# classes = ['???'] * model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name

# Define a list of colors for visualization

#load labels from csv
classes = ['???'] * 100

problems = []
problem_colors = []
arr_labels_ids = []

labels = pd.read_json('labels/nl.json')
arr_labels = labels.to_numpy()
rgb_current_color = '#000000'
# print(arr_labels[0][1])
# exit()



COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    current_color = arr_labels[class_id][1]
    rgb_current_color = tuple(int(current_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    color = rgb_current_color
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(arr_labels[class_id][0], obj['score'] * 100)
    cv2.putText(original_image_np, str(class_id), (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if(arr_labels[class_id][0] not in problems):
      problems.append(arr_labels[class_id][0])
      problem_colors.append('#%02x%02x%02x' % (color[0], color[1], color[2]))
      arr_labels_ids.append(class_id)
    

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8


def detect(image_path):
    INPUT_IMAGE_URL = image_path
    DETECTION_THRESHOLD = 0.25 

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        INPUT_IMAGE_URL, 
        interpreter, 
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection result
    print("Detection result:")
    # print(detection_result_image)
    #generate random string
    output_image_name = 'static/images/output/result-' + (uuid.uuid4().hex) + '.jpg'
    Image.fromarray(detection_result_image).save(output_image_name)
    return [output_image_name, problems, problem_colors,arr_labels_ids]