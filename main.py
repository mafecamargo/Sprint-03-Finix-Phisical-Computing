import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

MODEL_PATH = "emotion_rf.joblib"
FEATURE_LOG_CSV = "features_log.csv"
MAX_HISTORY = 15
FPS_SMOOTH = 0.9

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
    return np.array([[lm.x * w, lm.y * h, lm.z * max(w,h)] for lm in landmarks])

def interpupillary_distance(pts):
    return np.linalg.norm(pts[LM_LEFT_EYE][:2] - pts[LM_RIGHT_EYE][:2])

def pose_normalize(pts):
    # Centraliza e escala pelo interpupilar para tornar invariante à escala
    ipd = interpupillary_distance(pts)
    if ipd == 0:
        ipd = 1.0
    center = (pts[LM_LEFT_EYE][:2] + pts[LM_RIGHT_EYE][:2]) / 2
    norm = (pts[:, :2] - center) / ipd
    z = pts[:, 2:] / ipd
    return np.hstack([norm, z])

def extract_instant_features(pts):
    # pts: Nx3 array com coordenadas normalizadas (após pose_normalize)
    mouth_left = pts[LM_MOUTH_LEFT][:2]
    mouth_right = pts[LM_MOUTH_RIGHT][:2]
    mouth_top = pts[LM_MOUTH_TOP][:2]
    mouth_bottom = pts[LM_MOUTH_BOTTOM][:2]
    left_eyebrow = pts[LM_LEFT_EYEBROW][:2]
    right_eyebrow = pts[LM_RIGHT_EYEBROW][:2]
    left_eye = pts[LM_LEFT_EYE][:2]
    right_eye = pts[LM_RIGHT_EYE][:2]
    nose = pts[LM_NOSE][:2]

    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_open = np.linalg.norm(mouth_top - mouth_bottom)
    brow_left_h = np.linalg.norm(left_eyebrow - left_eye)
    brow_right_h = np.linalg.norm(right_eyebrow - right_eye)
    nose_mouth = np.linalg.norm(nose - mouth_top)

    # Relações e ângulos
    mouth_ratio = mouth_open / (eye_distance + 1e-6)
    mouth_width_rel = mouth_width / (eye_distance + 1e-6)
    brow_avg = (brow_left_h + brow_right_h) / 2 / (eye_distance + 1e-6)
    nose_mouth_rel = nose_mouth / (eye_distance + 1e-6)

    # ângulo das sobrancelhas (vetor)
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
        return np.zeros(5)  # mesma dimensão que instant_features

    # use até as últimas 5 frames
    recent = np.array(history_pts[-5:])  # shape: (frames, num_landmarks, 3)
    # diferenças entre frames (velocidade por coordenada)
    diffs = np.diff(recent, axis=0)     # shape: (frames-1, num_landmarks, 3)
    vel = diffs.mean(axis=0)            # shape: (num_landmarks, 3)
    acc = np.diff(diffs, axis=0).mean(axis=0) if diffs.shape[0] > 1 else np.zeros_like(vel)

    # normas agregadas para reduzir dimensionalidade
    vel_norm = np.linalg.norm(vel)      # escalar
    acc_norm = np.linalg.norm(acc)      # escalar

    # variações específicas (y coordinate) das landmarks de interesse ao longo das frames
    mouth_top_y = recent[:, LM_MOUTH_TOP, 1]
    left_eyebrow_y = recent[:, LM_LEFT_EYEBROW, 1]
    right_eyebrow_y = recent[:, LM_RIGHT_EYEBROW, 1]

    mouth_top_std = np.std(mouth_top_y)
    left_brow_std = np.std(left_eyebrow_y)
    right_brow_std = np.std(right_eyebrow_y)

    return np.array([vel_norm, acc_norm, mouth_top_std, left_brow_std, right_brow_std], dtype=float)

def make_feature_vector(instant_feats, temporal_feats):
    return np.concatenate([instant_feats, temporal_feats])

def heuristic_label_from_feats(fv):
    mouth_ratio, mouth_width_rel, brow_avg, nose_mouth_rel, brow_angle = fv[:5]
    # regras simples (ajustáveis)
    if brow_avg < 0.28 and mouth_width_rel < 0.55:
        return "raiva"
    if brow_avg > 0.45 and mouth_ratio < 0.08:
        return "tristeza"
    return "neutro"

def train_from_csv(csv_path, model_path=MODEL_PATH):
    # Espera CSV com colunas: feat0..featN,label
    df = pd.read_csv(csv_path)
    X = df[[c for c in df.columns if c.startswith("feat")]].values
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    print("Train score:", clf.score(X_train, y_train))
    print("Val score:", clf.score(X_val, y_val))
    dump(clf, model_path)
    print("Modelo salvo em", model_path)
    return clf

def main(use_model=True, log_features=False):
    clf = None
    if use_model and os.path.exists(MODEL_PATH):
        clf = load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        history_preds = deque(maxlen=MAX_HISTORY)
        history_pts = deque(maxlen=10)  # para features temporais
        last_time = time.time()
        # opcional: criar arquivo de log
        if log_features and not os.path.exists(FEATURE_LOG_CSV):
            hdr = ",".join([f"feat{i}" for i in range(10)]) + ",label\n"
            open(FEATURE_LOG_CSV, "w").write(hdr)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            label = "sem rosto"
            prob = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts_raw = to_xy_array(lm, w, h)  # Nx3
                pts_norm = pose_normalize(pts_raw)
                history_pts.append(pts_norm)

                # extrai features instantâneas
                inst = extract_instant_features(pts_norm)
                temp = compute_temporal_features(list(history_pts))
                fv = make_feature_vector(inst, temp)

                # grava features para dataset (opcional) com label vazio (para rotular depois)
                if log_features:
                    row = ",".join([f"{x:.6f}" for x in fv]) + ",\n"
                    with open(FEATURE_LOG_CSV, "a") as f:
                        f.write(row)

                # inferência com modelo treinado quando disponível
                if clf is not None:
                    probs = clf.predict_proba(fv.reshape(1, -1))[0]
                    classes = clf.classes_
                    idx = np.argmax(probs)
                    label = classes[idx]
                    prob = probs[idx]
                else:
                    label = heuristic_label_from_feats(fv)
                    prob = 1.0

                # suavização temporal: conserva últimas N predições com peso pela prob
                history_preds.append((label, prob))
                # computa label mais frequente ponderada por prob
                counts = {}
                for lab, p in history_preds:
                    counts[lab] = counts.get(lab, 0.0) + p
                label = max(counts.items(), key=lambda kv: kv[1])[0]
                prob = counts[label] / (len(history_preds) + 1e-6)

                # desenho utilitário
                for i, p in enumerate(pts_raw):
                    if i % 4 == 0:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 200, 0), -1)
                # desenha retângulo e texto
                cv2.rectangle(frame, (10, 10), (220, 70), (0,0,0), -1)
                cv2.putText(frame, f"Emotion: {label}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"P: {prob:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # mostra FPS aproximado
            now = time.time()
            fps = 1.0 / (now - last_time + 1e-6)
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

            cv2.imshow("Emotion MediaPipe Improved", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("t"):
                # tecla t: grava última feature com rótulo (interativo) - para construir dataset manualmente
                if len(history_pts) > 0:
                    # pede rótulo via terminal (rápido)
                    lbl = input("Digite rótulo para a última amostra (raiva/tristeza/neutro): ").strip()
                    inst = extract_instant_features(history_pts[-1])
                    temp = compute_temporal_features(list(history_pts))
                    fv = make_feature_vector(inst, temp)
                    with open(FEATURE_LOG_CSV, "a") as f:
                        row = ",".join([f"{x:.6f}" for x in fv]) + f",{lbl}\n"
                        f.write(row)
                    print("Amostra salva.")
            if key == ord("s"):
                # tecla s: salva modelo treinando com CSV se existir
                if os.path.exists(FEATURE_LOG_CSV):
                    try:
                        clf_new = train_from_csv(FEATURE_LOG_CSV, MODEL_PATH)
                        clf = clf_new
                    except Exception as e:
                        print("Erro ao treinar:", e)
                else:
                    print("Nenhum CSV de features encontrado:", FEATURE_LOG_CSV)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # use_model=True para tentar carregar modelo preexistente
    # log_features=False para não logar automaticamente (use True para coletar dados)
    main(use_model=True, log_features=False)
