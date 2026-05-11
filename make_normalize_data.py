import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


# create mediapipe Module
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# mediapipe object
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# start cv2 Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create list to save results
lm_results_list = []

# Create const Variable
no_of_frames = 600

def get_distance(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle(a, b, c):
    """Tính góc tại đỉnh b"""
    a = np.array(a);
    b = np.array(b);
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


def make_landmark_timestep(results):
    lm_list = []
    landmarks = results.pose_landmarks.landmark

    # --- BƯỚC 1: TRÍCH XUẤT CÁC ĐIỂM KEYPOINT ---
    # Gốc tọa độ (Anchor): Trung điểm hông
    hip_l = [landmarks[23].x, landmarks[23].y]
    hip_r = [landmarks[24].x, landmarks[24].y]
    center = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]

    # Các điểm phục vụ tính toán góc và khoảng cách
    shldr_l = [landmarks[11].x, landmarks[11].y];
    shldr_r = [landmarks[12].x, landmarks[12].y]
    elbow_l = [landmarks[13].x, landmarks[13].y];
    elbow_r = [landmarks[14].x, landmarks[14].y]
    wrist_l = [landmarks[15].x, landmarks[15].y];
    wrist_r = [landmarks[16].x, landmarks[16].y]
    knee_l = [landmarks[25].x, landmarks[25].y];
    knee_r = [landmarks[26].x, landmarks[26].y]
    ankle_l = [landmarks[27].x, landmarks[27].y];
    ankle_r = [landmarks[28].x, landmarks[28].y]
    mouth_l = [landmarks[9].x, landmarks[9].y];
    mouth_r = [landmarks[10].x, landmarks[10].y]
    mouth_center = [(mouth_l[0] + mouth_r[0]) / 2, (mouth_l[1] + mouth_r[1]) / 2]

    # --- BƯỚC 2: CHUẨN HÓA QUY MÔ (SCALING) ---
    # Dùng chiều dài xương đùi hoặc chiều dài thân để scale (rất ổn định)
    torso_size = get_distance([(shldr_l[0] + shldr_r[0]) / 2, (shldr_l[1] + shldr_r[1]) / 2], center)
    if torso_size == 0: torso_size = 1

    # --- BƯỚC 3: TRÍCH XUẤT TỌA ĐỘ NORMALIZE (132 features) ---
    for lm in landmarks:
        # Dời gốc về center và chia cho torso_size
        norm_x = (lm.x - center[0]) / torso_size
        norm_y = (lm.y - center[1]) / torso_size
        norm_z = lm.z / torso_size
        lm_list.extend([norm_x, norm_y, norm_z, lm.visibility])

    # --- BƯỚC 4: THÊM "GÓC" VÀ "KHOẢNG CÁCH"  ---
    # 1. Các góc quan trọng (Học tư thế: Đứng, Ngồi, Uống nước)
    angles = [
        calculate_angle(shldr_l, hip_l, knee_l),
        calculate_angle(shldr_r, hip_r, knee_r),
        calculate_angle(hip_l, knee_l, ankle_l),
        calculate_angle(hip_r, knee_r, ankle_r),
        calculate_angle(shldr_l, elbow_l, wrist_l),
        calculate_angle(shldr_r, elbow_r, wrist_r)
    ]

    # 2. Các khoảng cách tương đối (Học hành động: Vỗ tay, Uống nước)
    distances = [
        get_distance(wrist_l, wrist_r),  # Khoảng cách 2 tay (Vỗ tay)
        get_distance(wrist_l, mouth_center),  # Tay trái tới miệng (Uống nước)
        get_distance(wrist_r, mouth_center)  # Tay phải tới miệng (Uống nước)
    ]

    # Thêm vào list (Normalize về 0-1 cho góc và scale cho khoảng cách)
    lm_list.extend([a / 180.0 for a in angles])
    lm_list.extend([d / torso_size for d in distances])

    return lm_list

while len(lm_results_list) < no_of_frames:
    # Read frame
    ret, frameBGR = cap.read()

    # Exit Sign
    key = cv2.waitKey(1)
    if (key == 13) or (key == 27) or (not ret):
        break

    frameRGB = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)

    # Read landmarks in frame
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        # draw pose Connection in frame
        mp_draw.draw_landmarks(frameBGR, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Save pose landmarks results
        pose_lm = make_landmark_timestep(results)
        lm_results_list.append(pose_lm)

    cv2.imshow("Pose Detection Frame", frameBGR)

pose_data = pd.DataFrame(lm_results_list)
pose_data.to_csv("clapping.csv",index=False)
print("Data saved")


cap.release()
cv2.destroyAllWindows()
