import math
from typing import Tuple, Union
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def calculate_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = math.atan2(delta_y, delta_x)
    return angle * (180.0 / math.pi)


def crop_face(image_path):
    image = cv2.imread(image_path)
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        # To improve performance, optionally mark the image as not writeable tow
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                image_rows, image_cols, _ = image.shape
                left_eye = _normalized_to_pixel_coordinates(detection.location_data.relative_keypoints[0].x, detection.location_data.relative_keypoints[0].y,
                                                            image_cols, image_rows)
                right_eye = _normalized_to_pixel_coordinates(detection.location_data.relative_keypoints[1].x, detection.location_data.relative_keypoints[1].y,
                                                            image_cols, image_rows)
                if left_eye is not None and right_eye is not None:
                    angle = calculate_angle(left_eye, right_eye)
                    eyes_center = (
                        (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        eyes_center, angle, 1.0)
                    aligned = cv2.warpAffine(
                        image.copy(), rotation_matrix, (image_cols, image_rows), flags=cv2.INTER_LINEAR)
                else:
                    aligned = image.copy()
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                aligned_results = face_detection.process(aligned)
                aligned = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                if aligned_results.detections:
                    for aligned_detection in aligned_results.detections:
                        relative_bounding_box = aligned_detection.location_data.relative_bounding_box
                        rect_start_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                            image_rows)
                        rect_end_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin + relative_bounding_box.width,
                            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                            image_rows)
                        if rect_end_point is None or rect_start_point is None:
                            continue
                        face_image = aligned[rect_start_point[1]:rect_end_point[1],
                                            rect_start_point[0]:rect_end_point[0]]
                mp_drawing.draw_detection(image, detection)
