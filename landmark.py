import cv2
import numpy as np
import utils


def predict(image):
    # FACE DETECTION
    #image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haarcascade = "haarcascade_frontalface_alt2.xml"
    detector = cv2.CascadeClassifier(haarcascade)
    faces = detector.detectMultiScale(image_gray)
    
    for face in faces:
        (x, y, w, d) = face
        cv2.rectangle(image, (x, y), (x+w, y+d), (255, 255, 255), 2)

    # LANDMARK
    LBFmodel = "lbfmodel.yaml"
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(image_gray, faces)
    for landmark in landmarks:
        i = 0
        for x, y in landmark[0]:
            #cv2.putText(image, f'{i}', (int(x), int(y)),
            #            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)
            i += 1

    l_distance = utils.distance(landmarks[0][0][37], landmarks[0][0][40])
    l_midpoint = utils.midpoint(landmarks[0][0][37], landmarks[0][0][40])
    r_distance = utils.distance(landmarks[0][0][47], landmarks[0][0][44])
    r_midpoint = utils.midpoint(landmarks[0][0][47], landmarks[0][0][44])
    print(f'Left midpoint{l_midpoint}')
    for x, y in [l_midpoint, r_midpoint]:
        cv2.circle(image, (int(x), int(y)),  int(l_distance/2), (255, 0, 0), 1)
    
    # ROI
    # Gaze detection
    left_eye_region = np.array([(int(landmarks[0][0][36][0]), int(landmarks[0][0][36][1])),
                                (int(landmarks[0][0][37][0]),
                                 int(landmarks[0][0][37][1])),
                                (int(landmarks[0][0][38][0]),
                                 int(landmarks[0][0][38][1])),
                                (int(landmarks[0][0][39][0]),
                                 int(landmarks[0][0][39][1])),
                                (int(landmarks[0][0][40][0]),
                                 int(landmarks[0][0][40][1])),
                                (int(landmarks[0][0][41][0]), int(landmarks[0][0][41][1]))], np.int32)
    right_eye_region = np.array([(int(landmarks[0][0][43][0]), int(landmarks[0][0][43][1])),
                                (int(landmarks[0][0][44][0]),
                                 int(landmarks[0][0][44][1])),
                                (int(landmarks[0][0][45][0]),
                                 int(landmarks[0][0][45][1])),
                                (int(landmarks[0][0][46][0]),
                                 int(landmarks[0][0][46][1])),
                                (int(landmarks[0][0][47][0]),
                                 int(landmarks[0][0][47][1])),
                                (int(landmarks[0][0][42][0]), int(landmarks[0][0][42][1]))], np.int32)

    '''
    # eyes = [left_eye_region, right_eye_region]
    # height, width, _ = image.shape
    # mask = np.zeros((height, width), np.uint8)
    # rMask = mask.copy()

    # cv2.polylines(mask, [left_eye_region], True, 255, 2)
    # cv2.fillPoly(mask, [left_eye_region], 255)
    # left_eye = cv2.bitwise_and(image_gray, image_gray, mask=mask)

    # min_x = np.min(left_eye_region[:, 0])
    # max_x = np.max(left_eye_region[:, 0])
    # min_y = np.min(left_eye_region[:, 1])
    # max_y = np.max(left_eye_region[:, 1])
    # gray_eye = left_eye[min_y: max_y, min_x: max_x]
    # _, threshold_eye = cv2.threshold(gray_eye, 90, 255, cv2.THRESH_BINARY)

    # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    # eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    # cv2.imshow("Eye", eye)
    # cv2.imshow("Threshold", threshold_eye)
    # cv2.imshow("Left eye", left_eye)

    # cv2.imwrite('image.jpg',image)
    # cv2.imshow('',image)
    # cv2.waitKey(0)
    '''
    
    data = {
        'left': {
            'x': l_midpoint[0],
            'y': l_midpoint[1],
            'r': l_distance/2
        },
        'right': {
            'x': r_midpoint[0],
            'y': r_midpoint[1],
            'r': r_distance/2
        }
    }
    
    return(data, image)
#print(predict('./test_images/2.png'))
