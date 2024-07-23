import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

################################################
# visualise the bounding box
def visualize(
    image,
    detection_result
):
    """Draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """

    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # red

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name

        print(f"\r{category_name}", end="")

        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image
################################################

base_options = python.BaseOptions(
    model_asset_path="models/ssd_mobilenet_v2.tflite"
)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    max_results=5,
    score_threshold=0.3,
)
detector = vision.ObjectDetector.create_from_options(options)


## capturing video stream

capture = cv2.VideoCapture("test_media/jagriti_video1.mp4")
writer = cv2.VideoWriter(
    filename="test_media/jagriti_video1_ai.mp4",
    fourcc=cv2.VideoWriter().fourcc(*'mp4v'), 
    fps=30.0, frameSize=(1280, 720), isColor=True
)

while capture.isOpened():
    is_frame_correct, cv_frame = capture.read()

    if not is_frame_correct:
        print("Can't receive frame (stream ended?), exiting...")
        break
    
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_frame)
    detection_result = detector.detect(mp_frame)

    np_annotated_frame = visualize(np.copy(mp_frame.numpy_view()), detection_result)
    
    writer.write(np_annotated_frame)


# When everything done, release the capture
capture.release()
writer.release()
cv2.destroyAllWindows()