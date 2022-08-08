import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def resize_to_default(frame):
    h, w = frame.shape[:2]
    ratio = 480/h
    new_h = int(ratio*h)
    new_w = int(ratio*w)
    dim = (new_w, new_h)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def resize_to_hd(frame):
    h, w = frame.shape[:2]
    ratio = 720/h
    new_h = int(ratio*h)
    new_w = int(ratio*w)
    dim = (new_w, new_h)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def resize_to_small(frame):
    h, w = frame.shape[:2]
    ratio = 360/h
    new_h = int(ratio*h)
    new_w = int(ratio*w)
    dim = (new_w, new_h)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def detect_drowsiness(image):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # if not results.multi_face_landmarks:
        #     continue
        annotated_image = image.copy()
        image2 = np.zeros((720, 1280, 3), np.uint8)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=image2,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
              image=image2,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=image2,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_LEFT_EYE,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=image2,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=image2,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_LIPS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            me = mp_face_mesh.FACEMESH_LIPS
            le = mp_face_mesh.FACEMESH_LEFT_EYE
            re = mp_face_mesh.FACEMESH_RIGHT_EYE
            # print(len(me))
        cv2.imwrite('result.png', annotated_image)
    return annotated_image, image2, face_landmarks, me, le, re

def calculate_distance(point_start, point_end):
    start_x = point_start[0]
    start_y = point_start[1]
    end_x = point_end[0]
    end_y = point_end[1]
    distance = math.sqrt(((end_x - start_x)**2)+((end_y - start_y)**2))
    return distance

def calculate_MAR(x1, y1):
    distance_0 = calculate_distance((x1[19], y1[19]), (x1[26], y1[26]))
    distance_1 = calculate_distance((x1[2], y1[2]), (x1[17], y1[17]))
    distance_2 = calculate_distance((x1[9], y1[9]), (x1[1], y1[1]))
    mar = (distance_1+distance_2)/2/distance_0
    return mar

def calculate_EAR_eye_left(x1, y1):
    distance_0 = calculate_distance((x1[14], y1[14]), (x1[12], y1[12]))
    distance_1 = calculate_distance((x1[3], y1[3]), (x1[13], y1[13]))
    distance_2 = calculate_distance((x1[6], y1[6]), (x1[4], y1[4]))
    ear = (distance_1+distance_2)/2/distance_0
    return ear

def calculate_EAR_eye_right(x1, y1):
    distance_0 = calculate_distance((x1[12], y1[12]), (x1[15], y1[15]))
    distance_1 = calculate_distance((x1[13], y1[13]), (x1[4], y1[4]))
    distance_2 = calculate_distance((x1[6], y1[6]), (x1[8], y1[8]))
    ear = (distance_1+distance_2)/2/distance_0
    return ear

def drowsiness_pipeline(image):
    annotated_image, image2, face_landmarks, me, le, re = detect_drowsiness(image)
    landmarks = face_landmarks.landmark
    connect_lines_mouth = list(me)
    connect_lines_left_eye = list(le)
    connect_lines_right_eye = list(re)

    x0 = []
    y0 = []
    idx = 0
    n = []

    for (start_idx, end_idx) in connect_lines_mouth:
        point_start = landmarks[start_idx]
        x0.append(point_start.x*6-2.3)
        y0.append(point_start.y*11-4.3)
        n.append(idx)
        idx += 1

    x1 = []
    y1 = []
    idx = 0
    n = []

    for (start_idx, end_idx) in connect_lines_left_eye:
        point_start = landmarks[start_idx]
        x1.append(point_start.x*6-2.5)
        y1.append(point_start.y*35-17)
        n.append(idx)
        idx += 1

    x2 = []
    y2 = []
    idx = 0
    n = []

    for (start_idx, end_idx) in connect_lines_right_eye:
        point_start = landmarks[start_idx]
        # point_start = landmarks[start_idx]
        # point_end = landmarks[end_idx]
        x2.append(point_start.x*13-4)
        y2.append(point_start.y*60-29.8)
        n.append(idx)
        idx += 1

    ear_0 = calculate_MAR(x0, y0)
    ear_1 = calculate_EAR_eye_left(x1, y1)
    ear_2 = calculate_EAR_eye_right(x2, y2)

    ear_eye = (ear_1+ear_2)/2
    ear_mouth = ear_0
    
    is_closed = 0
    is_yawn = 0

    if ear_eye < 1:
        is_closed = 1
    else:
        is_closed = 0

    if ear_mouth > 0.5:
        is_yawn = 1
    else:
        is_yawn = 0
    
    return annotated_image, image2, ear_eye, ear_mouth, is_closed, is_yawn

def plot_text(image, ear_eye, ear_mouth, is_closed, is_yawn, th_frame_eye, th_frame_mouth):
    
    is_warning = 0
    
    if is_closed == 1 and th_frame_eye > 3:
        text_notif_eye = "Eye Closed? Yes"
        is_warning = 1
    elif is_closed == -1 and th_frame_eye > 3:
        is_warning = 2
        text_notif_eye = "Eye Closed? ---"
    elif is_closed == -1:
        text_notif_eye = "Eye Closed? ---"
    else:
        text_notif_eye = "Eye Closed? No"

    if is_yawn == 1 and th_frame_mouth > 3:
        text_notif_mouth = "Yawn? Yes"
        is_warning = 1
    elif is_yawn == -1 and th_frame_mouth > 3:
        is_warning = 2
        text_notif_mouth = "Yawn? ---"
    elif is_yawn == -1:
        text_notif_mouth = "Yawn? ---"
    else:
        text_notif_mouth = "Yawn? No"
        
    text_ear_eye = 'Eye [EAR] : '+str(float("{0:.4f}".format(ear_eye)))
    text_ear_mouth = 'Mouth [MAR] : '+str(float("{0:.4f}".format(ear_mouth)))

    cv2.putText(image, text_ear_eye, (10, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
    cv2.putText(image, text_ear_eye, (10, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 6)
    
    cv2.putText(image, text_notif_eye, (10,100),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
    cv2.putText(image, text_notif_eye, (10,100),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 6)
    
    cv2.putText(image, text_ear_mouth, (10, 150),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
    cv2.putText(image, text_ear_mouth, (10, 150),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 0, 125), 6)
    
    cv2.putText(image, text_notif_mouth, (10,200),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
    cv2.putText(image, text_notif_mouth, (10,200),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 0, 125), 6)
    
    if is_warning==1:
        # GPIO.output(buzzer, GPIO.LOW)
        cv2.putText(image, "****MENGANTUK!****************", (10,600), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 15)
        cv2.putText(image, "****MENGANTUK!****************", (10,600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 12)
    elif is_warning==2:
        # GPIO.output(buzzer, GPIO.LOW)
        cv2.putText(image, "**PANDANGAN TIDAK FOKUS!****************", (10,600), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 15)
        cv2.putText(image, "**PANDANGAN TIDAK FOKUS!****************", (10,600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 12)
    # else:
        # GPIO.output(buzzer, GPIO.HIGH)
    
    return image

