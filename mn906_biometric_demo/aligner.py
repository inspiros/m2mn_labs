import cv2
import numpy as np
import dlib


class Aligner:
    def __init__(self, face_architecture='models/deploy.prototxt.txt',
                 face_weights='models/res10_300x300_ssd_iter_140000.caffemodel',
                 face_confidence_threshold=0.7,
                 landmark_model='./models/shape_predictor_68_face_landmarks.dat'):

        self.np_template = np.float32([(0.194157, 0.16926692), (0.7888591, 0.15817115), (0.4949509, 0.5144414)])
        self.face_detector = cv2.dnn.readNetFromCaffe(face_architecture, face_weights)
        self.confidence_threshold = face_confidence_threshold
        self.landmark_detector = dlib.shape_predictor(landmark_model)
        self.outer_eyes_and_nose = np.array([36, 45, 33])

    def detect_faces(self, img):
        confidences = []
        h, w, _ = img.shape
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            endY = min(h, endY)
            endX = min(w, endX)
            startX = max(0, startX)
            startY = max(0, startY)
            if ((endY - startY) > 5) and ((endX - startX) > 5):
                faces.append(dlib.rectangle(startX, startY, endX, endY))
                confidences.append(confidence)

        return faces, confidences

    def dist(self, x, y):
        return (x[0] +- y[0]) ** 2 + (x[1] - y[1]) ** 2

    def get_face_center(self, face):
        center = face.center()
        return [center.x, center.y]

    def get_central_face(self, img):
        faces = self.detect_faces(img)
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        if len(faces) == 0:
            return None
        central_face = faces[0]
        min_dist = self.dist(center, self.get_face_center(central_face))
        for face in faces[1:]:
            c = self.get_face_center(face)
            dist = self.dist(c, center)
            if dist < min_dist:
                min_dist = dist
                central_face = face
        return [central_face]

    def get_landmarks(self, img, face):
        points = self.landmark_detector(img, face)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self, img, face, size=96, return_landmarks=False):
        if face is None:
            return cv2.resize(img, (size, size))
        landmarks = self.get_landmarks(img, face)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        H = cv2.getAffineTransform(landmarks[self.outer_eyes_and_nose], size * self.np_template)
        aligned_face = cv2.warpAffine(img, H, (size, size))
        if return_landmarks:
            return aligned_face, landmarks
        return aligned_face
