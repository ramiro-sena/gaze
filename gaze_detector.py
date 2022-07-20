import argparse
from skimage import io
import cv2 as cv
import json
import numpy as np
import mediapipe as mp


def create_json(json_filename: str, data: dict | str, error=None):
    if error:
        json_filename = "error.json"

    if not json_filename.endswith(".json"):
        json_filename = json_filename + ".json"

    with open(json_filename, "w", encoding="utf-8") as f:
        if error:
            return json.dump({"error": data}, f, ensure_ascii=False, indent=4)
        return json.dump(data, f, ensure_ascii=False, indent=4)


def url(filename: str):
    try:
        image = io.imread(filename)
    except:
        create_json(filename, "image-not-found", "error")
        raise argparse.ArgumentTypeError(
            "Image can not be found at: %s" % filename)

    return image


def detect(frame: np.ndarray):
    mp_face_mesh = mp.solutions.face_mesh

    LEFT_IRIS = range(474, 478)
    RIGHT_IRIS = range(469, 473)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])

            [l_cx, l_cy] = map(lambda x: int(x), [l_cx, l_cy])
            [r_cx, r_cy] = map(lambda x: int(x), [r_cx, r_cy])

            mean_diameter_of_irises = np.mean(
                [l_radius, r_radius]) * 2  # in pixels
            MEAN_HUMAN_IRISES = 11.7  # in milimeters

            pixels_to_milimeters_ratio = MEAN_HUMAN_IRISES / mean_diameter_of_irises

            distance_between_irises = np.sqrt(
                np.power(l_cx - r_cx, 2) + np.power(l_cy - r_cy, 2)
            )  # in pixels

            pupillary_distance = distance_between_irises * pixels_to_milimeters_ratio

            data = {
                'left': {
                    'gaze': {
                        'x': l_cx,
                        'y': l_cy,
                        'radius': l_radius
                    },
                    'l': {
                        'x': l_cx - 5 * l_radius,
                        'y': l_cy
                    },
                    'r':{
                        'x': l_cx + 3 * l_radius,
                        'y': l_cy
                    },
                    't':{
                        'x': l_cx,
                        'y': l_cy - l_radius
                    },
                    'b': {
                        'x': l_cx, 
                        'y': l_cy + 5 * l_radius
                    }
                },
                'right': {
                    'gaze': {
                        'x': r_cx,
                        'y': r_cy,
                        'radius': r_radius
                    },
                    'l': {
                        'x': r_cx - 3 * r_radius,
                        'y': r_cy
                    },
                    'r':{
                        'x': r_cx + 5 * r_radius,
                        'y': r_cy
                    },
                    't':{
                        'x': r_cx,
                        'y': r_cy - r_radius
                    },
                    'b': {
                        'x': r_cx, 
                        'y': r_cy + 5 * r_radius
                    }
                }
            }

            return(data, frame)

        return {"face-not-found"}



