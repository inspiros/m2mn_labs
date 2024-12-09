from typing import Sequence

import os
import cv2
import numpy as np
import scipy.spatial.distance as scidist
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime

import parameters as ops
from aligner import Aligner
from perf_counter import PerfCounter


def extract_rep(model, img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (96, 96)), 1 / 255, (96, 96))
    model.setInput(blob)
    return model.forward()


def draw_faces(img, face=(), color=(0, 255, 0)):
    if not isinstance(face, Sequence):
        face = [face]
    for face_i in face:
        startX, startY, endX, endY = face_i.left(), face_i.top(), face_i.right(), face_i.bottom()
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)


def draw_landmarks(img, landmarks, color=(0, 255, 0), pivot_inds=None):
    if not isinstance(landmarks, Sequence):
        landmarks = [landmarks]
    for landmarks_i in landmarks:
        if pivot_inds is not None:
            non_pivot_color = tuple(int(c * .75) for c in color)
            for i, (x, y) in enumerate(landmarks_i.astype(np.int64)):
                is_pivot = i in pivot_inds
                cv2.circle(img, (x, y), 2 if is_pivot else 1, color if is_pivot else non_pivot_color, -1)
        else:
            for x, y in landmarks_i.astype(np.int64):
                cv2.circle(img, (x, y), 1, color, -1)


def parse_args():
    parser = ArgumentParser('MN906 - Face Verification Demo')
    parser.add_argument('--input', default=0,
                        help='input video stream')
    parser.add_argument('--output', default='logs',
                        help='output dir')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='automatically save images')
    parser.add_argument('--show_fps', action='store_true', default=False,
                        help='measure and display fps')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialization
    aligner = Aligner(face_confidence_threshold=ops.face_confidence_threshold)
    model = cv2.dnn.readNetFromTorch(ops.face_reco_model)

    # Logging information
    log_folder = os.path.join(args.output, 'log_' + datetime.today().strftime('%Y_%m_%d_%H-%M-%S'))
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(log_folder)
    if args.save_images:
        log_folder_enrol = os.path.join(log_folder, 'enrol')
        log_folder_verif = os.path.join(log_folder, 'verif')
        os.makedirs(log_folder_enrol)
        os.makedirs(log_folder_verif)

    vid = cv2.VideoCapture(args.input)
    ret, frame = vid.read()
    h, w = frame.shape[:2]
    if args.show_fps:
        pipeline_pc, det_pc, align_pc = PerfCounter(momentum=0.9), PerfCounter(momentum=0.9), PerfCounter(momentum=0.9)
    else:
        pipeline_pc, det_pc, align_pc = nullcontext(), nullcontext(), nullcontext()

    frame_number = 0
    f = open(os.path.join(log_folder, 'results.csv'), 'w')
    sep = ','
    f.write(f'frame_number{sep}face_id{sep}face_confidence_score{sep}face_verification_score{sep}verification_decision\n')
    enrol_reprs = ()
    while True:
        frame_number += 1
        ret, frame = vid.read()
        if not ret:
            continue
        with pipeline_pc:
            with det_pc:
                faces, confidences = aligner.detect_faces(frame)
            aligned_imgs, landmarks_list, new_reprs = [], [], []
            frame1 = frame.copy()
            if not len(enrol_reprs):
                draw_faces(frame1, faces)
            else:
                with align_pc:
                    for index in range(len(faces)):
                        face = faces[index]
                        aligned_img, landmarks = aligner.align(frame, face, return_landmarks=True)
                        new_repr = extract_rep(model, aligned_img)
                        aligned_imgs.append(aligned_img)
                        landmarks_list.append(landmarks)
                        new_reprs.append(new_repr)
                for index in range(len(faces)):
                    face = faces[index]
                    confidence = confidences[index]
                    aligned_img = aligned_imgs[index]
                    landmarks = landmarks_list[index]
                    new_repr = new_reprs[index]
                    d = scidist.cdist(enrol_reprs, new_repr, metric='euclidean').min()
                    verif_score = (2. - d) / 2.
                    startX, startY, endX, endY = face.left(), face.top(), face.right(), face.bottom()
                    if verif_score >= ops.threshold:
                        if (ops.cont_enroll and
                            len(enrol_reprs) < ops.number_of_templates and
                            verif_score >= ops.upper_enrolment_update_threshold and
                            verif_score <= ops.upper_enrolment_update_threshold):  # continous enrolment
                            enrol_reprs = np.concatenate([enrol_reprs, new_repr], axis=0)
                        text = f'User-1, conf: {confidence:.2f}, score: {verif_score:.2f}'
                        draw_landmarks(frame1, landmarks, color=(0, 255, 0), pivot_inds=aligner.outer_eyes_and_nose)
                        draw_faces(frame1, face, color=(0, 255, 0))
                        cv2.putText(frame1, text, (startX, startY - 10 if startY - 10 > 10 else startY + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 255, 0), 1)
                    else:
                        text = f'Unknown, conf: {confidence:.2f}, score: {verif_score:.2f}'
                        draw_landmarks(frame1, landmarks, color=(0, 0, 255), pivot_inds=aligner.outer_eyes_and_nose)
                        draw_faces(frame1, face, color=(0, 0, 255))
                        cv2.putText(frame1, text, (startX, startY - 10 if startY - 10 > 10 else startY + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 0, 255), 1)
                    f.write(f'{frame_number}{sep}{index}{sep}{confidence:.05f}{sep}{verif_score:.05f}{sep}{verif_score >= ops.threshold}\n')
                    if args.save_images:
                        cv2.imwrite(
                            os.path.join(log_folder_verif, f'frame_{frame_number:06d}-face_{index}-aligned.jpg'), aligned_img)
                        cv2.imwrite(
                            os.path.join(log_folder_verif, f'frame_{frame_number:06d}-face_{index}.jpg'), frame[startY:endY, startX:endX])

            cv2.putText(frame1, f'#Enrolments: {len(enrol_reprs)}',
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if args.show_fps:
                cv2.putText(frame1, f'fps={pipeline_pc.fps:.02f} (detection_fps={det_pc.fps:.02f}, alignment_fps={align_pc.fps:.02f})',
                            (w - 360, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow('Face Verification', frame1)
        key = cv2.waitKey(1)
        if key == ord('q'):  # quit
            break
        elif key == ord('s'):  # save current frame
            cv2.imwrite(
                os.path.join(log_folder, f'frame_{frame_number:06d}-num-faces_{len(faces)}.jpg'), frame1)
            for index in range(len(aligned_imgs)):
                face = faces[index]
                aligned_img = aligner.align(frame, face, size=128)
                aligned_img1 = aligner.align(frame1, face, size=128)
                startX, startY, endX, endY = face.left(), face.top(), face.right(), face.bottom()
                cv2.imwrite(
                    os.path.join(log_folder, f'frame_{frame_number:06d}-face_{index}-aligned.jpg'), aligned_img)
                cv2.imwrite(
                    os.path.join(log_folder, f'frame_{frame_number:06d}-face_{index}-aligned-with-landmarks.jpg'), aligned_img1)
                cv2.imwrite(
                    os.path.join(log_folder, f'frame_{frame_number:06d}-face_{index}.jpg'), frame[startY:endY, startX:endX])
                cv2.imwrite(
                    os.path.join(log_folder, f'frame_{frame_number:06d}-face_{index}-with-landmarks.jpg'), frame1[startY:endY, startX:endX])
        elif key == ord(' '):  # enroll new face
            if len(faces) == 0:
                print('No face to be enrolled')
            elif len(faces) > 1:
                print('Too many faces')
            else:
                face = faces[0]
                if not len(enrol_reprs):
                    aligned_img = aligner.align(frame, face)
                    enrol_reprs = extract_rep(model, aligned_img)
                else:
                    # new_repr already computed
                    enrol_reprs = np.concatenate([enrol_reprs, new_reprs[0]])
                if args.save_images:
                    startX, startY, endX, endY = face.left(), face.top(), face.right(), face.bottom()
                    cv2.imwrite(
                        os.path.join(log_folder_enrol, f'enrol_frame_{len(enrol_reprs)}-aligned-face.jpg'), aligned_imgs[0])
                    cv2.imwrite(
                        os.path.join(log_folder_enrol, f'enrol_frame_{len(enrol_reprs)}-face.jpg'), frame[startY:endY, startX:endX])
    cv2.destroyAllWindows()
    vid.release()
    f.close()


if __name__ == '__main__':
    main()
