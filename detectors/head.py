# import cv2, math, numpy as np
# import mediapipe as mp

# # Initialize MediaPipe Face Mesh with iris landmarks refinement
# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # 3D model points of facial landmarks (in mm)
# MODEL_POINTS = np.array([
#     (0.0,    0.0,    0.0),       # Nose tip
#     (0.0,  -330.0,  -65.0),       # Chin
#     (-225.0, 170.0, -135.0),      # Left eye left corner
#     (225.0,  170.0, -135.0),      # Right eye right corner
#     (-150.0,-150.0, -125.0),      # Left mouth corner
#     (150.0, -150.0, -125.0)       # Right mouth corner
# ], dtype=np.float64)

# # Landmark IDs for the six points above
# LANDMARK_IDS = {
#     'nose_tip': 1,
#     'chin': 152,
#     'left_eye_outer': 33,
#     'right_eye_outer': 263,
#     'left_mouth': 61,
#     'right_mouth': 291
# }

# # Iris landmark indices from MediaPipe face mesh
# LEFT_IRIS_IDX  = [468, 469, 470, 471]
# RIGHT_IRIS_IDX = [473, 474, 475, 476]


# def get_image_points(landmarks, w: int, h: int) -> np.ndarray:
#     """
#     Extracts the 2D image points for head-pose estimation.
#     landmarks: list of normalized (x,y) tuples
#     returns: 6x2 array of pixel coordinates
#     """
#     return np.array([
#         (landmarks[LANDMARK_IDS['nose_tip']].x * w,
#          landmarks[LANDMARK_IDS['nose_tip']].y * h),
#         (landmarks[LANDMARK_IDS['chin']].x * w,
#          landmarks[LANDMARK_IDS['chin']].y * h),
#         (landmarks[LANDMARK_IDS['left_eye_outer']].x * w,
#          landmarks[LANDMARK_IDS['left_eye_outer']].y * h),
#         (landmarks[LANDMARK_IDS['right_eye_outer']].x * w,
#          landmarks[LANDMARK_IDS['right_eye_outer']].y * h),
#         (landmarks[LANDMARK_IDS['left_mouth']].x * w,
#          landmarks[LANDMARK_IDS['left_mouth']].y * h),
#         (landmarks[LANDMARK_IDS['right_mouth']].x * w,
#          landmarks[LANDMARK_IDS['right_mouth']].y * h)
#     ], dtype=np.float64)


# def solve_head_pose(image_points: np.ndarray, w: int, h: int):
#     """
#     Solves PnP to find rotation and translation vectors.
#     returns success flag, rvec, tvec.
#     """
#     focal_length = w
#     center = (w / 2, h / 2)
#     camera_matrix = np.array([
#         [focal_length, 0,            center[0]],
#         [0,            focal_length, center[1]],
#         [0,            0,            1]
#     ], dtype=np.float64)
#     dist_coeffs = np.zeros((4, 1))
#     success, rvec, tvec = cv2.solvePnP(
#         MODEL_POINTS,
#         image_points,
#         camera_matrix,
#         dist_coeffs,
#         flags=cv2.SOLVEPNP_ITERATIVE
#     )
#     return success, rvec, tvec


# def rotation_matrix_to_euler(R: np.ndarray):
#     """
#     Converts rotation matrix to Euler angles: (pitch, yaw, roll) in degrees.
#     """
#     sy = math.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
#     singular = sy < 1e-6

#     if not singular:
#         x = math.atan2(R[2,1], R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else:
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0

#     return math.degrees(x), math.degrees(y), math.degrees(z)


# def get_iris_centers(landmarks, w: int, h: int) -> dict:
#     """
#     Computes the pixel center of left and right iris by averaging their landmarks.
#     """
#     left_pts  = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_IRIS_IDX])
#     right_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_IRIS_IDX])
#     return {
#         'left_iris':  left_pts.mean(axis=0),
#         'right_iris': right_pts.mean(axis=0)
#     }


# def analyze_head(frame,
#                  yaw_thresh=10,
#                  pitch_thresh=15):
#     """
#     Returns:
#       { yaw, pitch, gaze, alert: bool }
#     """
#     h, w = frame.shape[:2]
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     res = face_mesh.process(rgb)

#     yaw = pitch = roll = 0.0
#     gaze = "Center"
#     alert = False

#     if res.multi_face_landmarks:
#         lm = res.multi_face_landmarks[0].landmark

#         # head-pose
#         ok, rvec, _ = solve_head_pose(get_image_points(lm, w, h), w, h)
#         if ok:
#             R, _ = cv2.Rodrigues(rvec)
#             pitch, yaw, roll = rotation_matrix_to_euler(R)

#         # gaze detection (left eye)
#         iris = get_iris_centers(lm, w, h)['left_iris']
#         eye_out = np.array((lm[33].x * w, lm[33].y * h))
#         eye_in  = np.array((lm[133].x * w, lm[133].y * h))
#         ctr     = (eye_out + eye_in) / 2
#         gx, gy  = iris - ctr
#         if abs(gy) > abs(gx) and abs(gy) > 5:
#             gaze = 'Down' if gy > 0 else 'Up'
#         elif abs(gx) > 5:
#             gaze = 'Right' if gx > 0 else 'Left'

#         # combine into alert
#         alert = abs(yaw) > yaw_thresh or abs(pitch) > pitch_thresh

#     return {
#         'yaw':   yaw,
#         'pitch': pitch,
#         'gaze':  gaze,
#         'alert': alert
#     }



import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh with iris landmarks refinement
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices
LEFT_IRIS_IDX  = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

def get_iris_centers(landmarks, w: int, h: int) -> dict:
    """
    Computes the pixel center of left and right iris by averaging their landmarks.
    """
    left_pts  = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_IRIS_IDX])
    right_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_IRIS_IDX])
    return {
        'left_iris':  left_pts.mean(axis=0),
        'right_iris': right_pts.mean(axis=0)
    }

def analyze_head(frame, movement_thresh=5):
    """
    Returns:
      { gaze: 'Center'|'Left'|'Right'|'Up'|'Down', alert: bool }
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze = "Center"
    alert = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # gaze detection (left eye)
        iris = get_iris_centers(lm, w, h)['left_iris']
        eye_out = np.array((lm[33].x * w, lm[33].y * h))
        eye_in  = np.array((lm[133].x * w, lm[133].y * h))
        ctr     = (eye_out + eye_in) / 2
        dx, dy  = iris - ctr

        if abs(dy) > abs(dx) and abs(dy) > movement_thresh:
            gaze = 'Down' if dy > 0 else 'Up'
        elif abs(dx) > movement_thresh:
            gaze = 'Right' if dx > 0 else 'Left'

        alert = gaze != "Center"

    return {
        'gaze':  gaze,
        'alert': alert
    }
