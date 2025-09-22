import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
import os
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh


MODEL_PATH = "emotion_rf.joblib"
FEATURE_LOG_CSV = "features_log.csv"
PREDICTION_CSV = "predictions_video.csv"
MAX_HISTORY = 15

# Índices úteis do Face Mesh (MediaPipe)
LM_LEFT_EYE = 33
LM_RIGHT_EYE = 263
LM_NOSE = 1
LM_MOUTH_LEFT = 61
LM_MOUTH_RIGHT = 291
LM_MOUTH_TOP = 13
LM_MOUTH_BOTTOM = 14
LM_LEFT_EYEBROW = 70
LM_RIGHT_EYEBROW = 300

def to_xy_array(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h, lm.z * max(w, h)] for lm in landmarks])

def interpupillary_distance(pts):
    return np.linalg.norm(pts[LM_LEFT_EYE][:2] - pts[LM_RIGHT_EYE][:2])

def pose_normalize(pts):
    ipd = interpupillary_distance(pts)
    if ipd == 0:
        ipd = 1.0
    center = (pts[LM_LEFT_EYE][:2] + pts[LM_RIGHT_EYE][:2]) / 2
    norm_xy = (pts[:, :2] - center) / ipd
    z = pts[:, 2:] / ipd
    return np.hstack([norm_xy, z])

def extract_instant_features(pts):
    mouth_left = pts[LM_MOUTH_LEFT][:2]
    mouth_right = pts[LM_MOUTH_RIGHT][:2]
    mouth_top = pts[LM_MOUTH_TOP][:2]
    mouth_bottom = pts[LM_MOUTH_BOTTOM][:2]
    left_eyebrow = pts[LM_LEFT_EYEBROW][:2]
    right_eyebrow = pts[LM_RIGHT_EYEBROW][:2]
    left_eye = pts[LM_LEFT_EYE][:2]
    right_eye = pts[LM_RIGHT_EYE][:2]
    nose = pts[LM_NOSE][:2]

    eye_distance = np.linalg.norm(left_eye - right_eye) + 1e-6
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_open = np.linalg.norm(mouth_top - mouth_bottom)
    brow_left_h = np.linalg.norm(left_eyebrow - left_eye)
    brow_right_h = np.linalg.norm(right_eyebrow - right_eye)
    nose_mouth = np.linalg.norm(nose - mouth_top)

    mouth_ratio = mouth_open / eye_distance
    mouth_width_rel = mouth_width / eye_distance
    brow_avg = (brow_left_h + brow_right_h) / 2 / eye_distance
    nose_mouth_rel = nose_mouth / eye_distance

    brow_vec = (right_eyebrow - left_eyebrow)
    brow_angle = np.arctan2(brow_vec[1], brow_vec[0])

    feats = np.array([
        mouth_ratio,
        mouth_width_rel,
        brow_avg,
        nose_mouth_rel,
        brow_angle
    ], dtype=float)
    return feats

def compute_temporal_features(history_pts):
    # history_pts: lista de arrays (num_landmarks, 3), mais recente último
    n = len(history_pts)
    if n < 3:
        return np.zeros(5, dtype=float)
    recent = np.array(history_pts[-5:])  # shape: (frames, num_landmarks, 3)
    diffs = np.diff(recent, axis=0)      # (frames-1, num_landmarks, 3)
    vel = diffs.mean(axis=0)             # (num_landmarks, 3)
    acc = np.diff(diffs, axis=0).mean(axis=0) if diffs.shape[0] > 1 else np.zeros_like(vel)
    vel_norm = np.linalg.norm(vel)
    acc_norm = np.linalg.norm(acc)

    mouth_top_y = recent[:, LM_MOUTH_TOP, 1]
    left_eyebrow_y = recent[:, LM_LEFT_EYEBROW, 1]
    right_eyebrow_y = recent[:, LM_RIGHT_EYEBROW, 1]

    mouth_top_std = float(np.std(mouth_top_y))
    left_brow_std = float(np.std(left_eyebrow_y))
    right_brow_std = float(np.std(right_eyebrow_y))

    return np.array([vel_norm, acc_norm, mouth_top_std, left_brow_std, right_brow_std], dtype=float)

def make_feature_vector(instant_feats, temporal_feats):
    return np.concatenate([instant_feats, temporal_feats])

def heuristic_label_from_feats(fv):
    mouth_ratio, mouth_width_rel, brow_avg = fv[0], fv[1], fv[2]
    if brow_avg < 0.28 and mouth_width_rel < 0.55:
        return "raiva"
    if brow_avg > 0.45 and mouth_ratio < 0.08:
        return "tristeza"
    return "neutro"

def train_from_csv(csv_path, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)
    X = df[[c for c in df.columns if c.startswith("feat")]].values
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    print("Train score:", clf.score(X_train, y_train))
    print("Val score:", clf.score(X_val, y_val))
    dump(clf, model_path)
    print("Modelo salvo em", model_path)
    return clf

def process_video(video_path,
                  use_model=True,
                  log_features=False,
                  save_predictions=True,
                  skip_every=1,
                  resize_factor=1.0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    clf = None
    if use_model and os.path.exists(MODEL_PATH):
        clf = load(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_ms = int(1000 / fps)

    if save_predictions:
        pd.DataFrame(columns=["frame", "time_s", "label", "prob"]).to_csv(PREDICTION_CSV, index=False)

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        history_preds = deque(maxlen=MAX_HISTORY)
        history_pts = deque(maxlen=10)
        frame_idx = 0
        last_time = time.time()

        if log_features and not os.path.exists(FEATURE_LOG_CSV):
            hdr = ",".join([f"feat{i}" for i in range(10)]) + ",label\n"
            open(FEATURE_LOG_CSV, "w").write(hdr)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % skip_every != 0:
                continue

            if resize_factor != 1.0:
                frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            label = "sem rosto"
            prob = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts_raw = to_xy_array(lm, w, h)
                pts_norm = pose_normalize(pts_raw)
                history_pts.append(pts_norm)

                inst = extract_instant_features(pts_norm)
                temp = compute_temporal_features(list(history_pts))
                fv = make_feature_vector(inst, temp)

                if log_features:
                    # grava como feat0..feat9,label (label vazio para rotular depois)
                    row = {f"feat{i}": float(fv[i]) for i in range(len(fv))}
                    row["label"] = ""
                    pd.DataFrame([row]).to_csv(FEATURE_LOG_CSV, mode="a", header=False, index=False)

                if clf is not None:
                    probs = clf.predict_proba(fv.reshape(1, -1))[0]
                    classes = clf.classes_
                    idx = int(np.argmax(probs))
                    label = classes[idx]
                    prob = float(probs[idx])
                else:
                    label = heuristic_label_from_feats(fv)
                    prob = 1.0

                history_preds.append((label, prob))
                counts = {}
                for lab, p in history_preds:
                    counts[lab] = counts.get(lab, 0.0) + p
                label = max(counts.items(), key=lambda kv: kv[1])[0]
                prob = counts[label] / (len(history_preds) + 1e-6)

                for i, p in enumerate(pts_raw):
                    if i % 4 == 0:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 200, 0), -1)

                cv2.rectangle(frame, (10, 10), (260, 70), (0,0,0), -1)
                cv2.putText(frame, f"Emotion: {label}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"P: {prob:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # salvar predições por frame
            if save_predictions:
                timestamp = frame_idx / fps
                pd.DataFrame([{"frame": frame_idx, "time_s": timestamp, "label": label, "prob": prob}])\
                    .to_csv(PREDICTION_CSV, mode="a", header=False, index=False)

            now = time.time()
            fps_shown = 1.0 / (now - last_time + 1e-6)
            last_time = now
            cv2.putText(frame, f"FPS: {fps_shown:.1f}", (w-140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

            cv2.imshow("Emotion MediaPipe Video", frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27:
                break
            if key == ord("s"):
                # tecla s: treinar modelo a partir do CSV de features, se existir
                if os.path.exists(FEATURE_LOG_CSV):
                    try:
                        clf = train_from_csv(FEATURE_LOG_CSV, MODEL_PATH)
                    except Exception as e:
                        print("Erro ao treinar:", e)
                else:
                    print("Nenhum CSV de features encontrado:", FEATURE_LOG_CSV)

    cap.release()
    cv2.destroyAllWindows()
    print("Processamento finalizado. Predições salvas em", PREDICTION_CSV if save_predictions else "nenhum arquivo")

if __name__ == "__main__":
    # Ajuste aqui:
    video_path = "videoplayback.mp4"
    use_model = True
    log_features = False
    save_predictions = True
    skip_every = 1
    resize_factor = 1.0

    process_video(video_path,
                  use_model=use_model,
                  log_features=log_features,
                  save_predictions=save_predictions,
                  skip_every=skip_every,
                  resize_factor=resize_factor)
