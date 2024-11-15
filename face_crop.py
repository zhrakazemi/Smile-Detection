import math
from typing import Tuple, Union
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


 class FaceAligner:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        """
        Initialize the FaceAligner with the specified model selection and confidence.
        """
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=self.model_selection, 
            min_detection_confidence=self.min_detection_confidence
        )

    @staticmethod
    def _normalized_to_pixel_coordinates(
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    @staticmethod
    def calculate_angle(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate the angle between two points in degrees."""
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]
        angle = math.atan2(delta_y, delta_x)
        return angle * (180.0 / math.pi)

    def align_face(self, image_path: str) -> Union[None, Tuple[cv2.Mat, cv2.Mat]]:
        """Align and crop face from the given image."""
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found.")
            return None

        image_rows, image_cols, _ = image.shape
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self.face_detection.process(image_rgb)

        if not results.detections:
            print("No face detected.")
            return None

        # Align the face based on eye coordinates
        for detection in results.detections:
            left_eye = self._normalized_to_pixel_coordinates(
                detection.location_data.relative_keypoints[0].x,
                detection.location_data.relative_keypoints[0].y,
                image_cols, image_rows
            )
            right_eye = self._normalized_to_pixel_coordinates(
                detection.location_data.relative_keypoints[1].x,
                detection.location_data.relative_keypoints[1].y,
                image_cols, image_rows
            )
            if left_eye is not None and right_eye is not None:
                angle = self.calculate_angle(left_eye, right_eye)
                eyes_center = (
                    (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2
                )
                rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
                aligned_image = cv2.warpAffine(
                    image, rotation_matrix, (image_cols, image_rows), flags=cv2.INTER_LINEAR
                )
            else:
                aligned_image = image.copy()

            # Convert the aligned image to RGB for detection
            aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
            aligned_results = self.face_detection.process(aligned_image_rgb)

            # Detect the face bounding box in the aligned image
            if aligned_results.detections:
                for aligned_detection in aligned_results.detections:
                    relative_bounding_box = aligned_detection.location_data.relative_bounding_box
                    rect_start_point = self._normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin, relative_bounding_box.ymin,
                        image_cols, image_rows
                    )
                    rect_end_point = self._normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin + relative_bounding_box.width,
                        relative_bounding_box.ymin + relative_bounding_box.height,
                        image_cols, image_rows
                    )

                    if rect_start_point and rect_end_point:
                        # Crop the face from the aligned image
                        face_image = aligned_image[rect_start_point[1]:rect_end_point[1],
                                                   rect_start_point[0]:rect_end_point[0]]
                        return face_image
                
        return None