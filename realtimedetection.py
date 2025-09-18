import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import time
import pandas as pd

st.set_page_config(page_title="Emotion Detector (Smooth + Grad-CAM)", layout="wide")
st.title("😃 Real-Time Emotion Detector with Grad-CAM — Smooth Mode")
st.write("Detects emotions from webcam feed, shows **probabilities**, and overlays **Grad-CAM**.")

# ----------------- Sidebar controls -----------------
MODE            = st.sidebar.radio("Camera Mode", ["Local Webcam", "Browser Camera"])
TARGET_FPS      = st.sidebar.slider("Target FPS (local webcam only)", 5, 30, 20, 1)
GRADCAM_EVERY   = st.sidebar.slider("Compute Grad-CAM every N frames", 1, 10, 5, 1)
CHART_EVERY     = st.sidebar.slider("Update chart every N frames", 1, 20, 10, 1)
FRAME_WIDTH     = st.sidebar.selectbox("Capture width", [320, 480, 640, 800, 960, 1280], index=2)
FRAME_HEIGHT    = st.sidebar.selectbox("Capture height", [240, 360, 480, 600, 720, 960], index=2)
SHOW_HEATMAP    = st.sidebar.toggle("Show Grad-CAM overlay", True)

# ----------------- Load Model -----------------
@st.cache_resource
def load_cam_model():
    with open("emotiondetector.json", "r") as f:
        model_json = f.read()
    seq_model = model_from_json(model_json)
    seq_model.load_weights("emotiondetector.h5")

    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = inputs
    last_conv_act = None
    last_conv_name = None
    for layer in seq_model.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_act = x
            last_conv_name = layer.name

    if last_conv_act is None:
        raise ValueError("No Conv2D layer found — Grad-CAM needs at least one Conv2D layer.")

    cam_model = tf.keras.Model(inputs=inputs, outputs=[last_conv_act, x], name="cam_model")
    _ = cam_model.predict(np.zeros((1, 48, 48, 1), dtype=np.float32), verbose=0)
    return cam_model, last_conv_name

cam_model, last_conv_name = load_cam_model()
st.sidebar.caption(f"Grad-CAM layer: `{last_conv_name}`")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# ----------------- Utils -----------------
def extract_features(gray48):
    arr = np.asarray(gray48, dtype=np.float32) / 255.0
    return arr.reshape(1, 48, 48, 1)

def to1d(arr):
    a = np.asarray(arr)
    a = np.squeeze(a)
    return a.astype(np.float32)

def compute_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(img_array)
        top_idx = tf.argmax(preds[0])
        score = preds[:, top_idx]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), to1d(preds)

def overlay_gradcam(heatmap, bgr_image, alpha=0.45, colormap=cv2.COLORMAP_JET):
    hm = cv2.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]))
    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, colormap)
    return cv2.addWeighted(bgr_image, 1 - alpha, hm, alpha, 0)

# ----------------- Placeholders -----------------
FRAME = st.empty()
st.sidebar.title("Now")
emo_ph   = st.sidebar.empty()
conf_ph  = st.sidebar.empty()
chart_ph = st.sidebar.empty()

# ----------------- Local Webcam Mode -----------------
if MODE == "Local Webcam":
    run = st.checkbox("Start Webcam (local machine only)")
    if run:
        cv2.setUseOptimized(True)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(FRAME_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("Webcam not found. Try running locally with a camera.")
        else:
            haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            ema_preds = np.ones(7, dtype=np.float32) / 7.0
            ema_alpha = 0.2
            frame_idx = 0
            target_dt = 1.0 / max(1, TARGET_FPS)

            try:
                while True:
                    t0 = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = haar.detectMultiScale(gray, 1.3, 5)
                    do_cam = SHOW_HEATMAP and (frame_idx % GRADCAM_EVERY == 0)

                    if len(faces) > 0:
                        for (x, y, w, h) in faces[:3]:
                            face = gray[y:y+h, x:x+w]
                            face48 = cv2.resize(face, (48, 48))
                            img_in = extract_features(face48)

                            if do_cam:
                                heatmap, preds = compute_gradcam(img_in)
                            else:
                                _, preds = cam_model.predict(img_in, verbose=0)
                                preds = to1d(preds)
                                heatmap = None

                            ema_preds = to1d((1 - ema_alpha) * ema_preds + ema_alpha * preds)
                            cls = int(np.argmax(ema_preds))
                            emo = labels[cls]
                            conf = float(np.round(ema_preds[cls] * 100.0, 2))

                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, f"{emo} ({conf}%)", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            if SHOW_HEATMAP and heatmap is not None:
                                face_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                                face_cam = overlay_gradcam(heatmap, face_bgr)
                                frame[y:y+h, x:x+w] = cv2.resize(face_cam, (w, h))

                            emo_ph.markdown(f"### Emotion: **{emo}**")
                            conf_ph.markdown(f"### Confidence: **{conf}%**")

                        if frame_idx % CHART_EVERY == 0:
                            df = pd.DataFrame(
                                {"probability": ema_preds},
                                index=[labels[i] for i in range(len(labels))]
                            )
                            chart_ph.bar_chart(df, use_container_width=True)

                    FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    dt = time.perf_counter() - t0
                    if dt < target_dt:
                        time.sleep(target_dt - dt)
                    frame_idx += 1
            finally:
                cap.release()
    else:
        st.info("Toggle **Start Webcam** to begin (only works locally).")

# ----------------- Browser Camera Mode -----------------
else:
    st.info("Use your browser’s camera to take snapshots (works on Streamlit Cloud).")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Convert buffer → OpenCV image
        bytes_data = img_file_buffer.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = haar.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces[:3]:
                face = gray[y:y+h, x:x+w]
                face48 = cv2.resize(face, (48, 48))
                img_in = extract_features(face48)

                heatmap, preds = compute_gradcam(img_in)
                cls = int(np.argmax(preds))
                emo = labels[cls]
                conf = float(np.round(preds[cls] * 100.0, 2))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emo} ({conf}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if SHOW_HEATMAP and heatmap is not None:
                    face_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                    face_cam = overlay_gradcam(heatmap, face_bgr)
                    frame[y:y+h, x:x+w] = cv2.resize(face_cam, (w, h))

                emo_ph.markdown(f"### Emotion: **{emo}**")
                conf_ph.markdown(f"### Confidence: **{conf}%**")

            df = pd.DataFrame(
                {"probability": preds},
                index=[labels[i] for i in range(len(labels))]
            )
            chart_ph.bar_chart(df, use_container_width=True)

        FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
