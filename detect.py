import cv2
from face_crop import FaceAligner
import numpy as np
import tensorflow as tf


face_aligner = FaceAligner()
model = tf.keras.models.load_model('smile-detection.keras')


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_image = face_aligner.align_face(image=image)
    face = tf.image.resize(face_image, [224, 224])
    face = np.expand_dims(face, axis=0)

    predict = tf.where(model.predict(face) < 0.5, 0, 1)
    cv2.putText(image, 'smile' if predict == 1 else 'non smile', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Smile Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()