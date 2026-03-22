import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Virtual Makeup", layout="wide")

st.title("💄 Virtual Lipstick Try-On App")

# Load MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Lipstick shades
lip_colors = {
    "Red ❤️": (0, 0, 255),
    "Pink 💗": (147, 20, 255),
    "Nude 🤎": (180, 140, 120),
    "Plum 🍇": (80, 0, 120)
}

# Lip landmark indices (MediaPipe)
LIP_IDS = [61,146,91,181,84,17,314,405,321,375,291]

# Function to apply lipstick
def apply_lipstick(image, landmarks, color):
    lips_points = []

    for i in LIP_IDS:
        x = int(landmarks[i].x * image.shape[1])
        y = int(landmarks[i].y * image.shape[0])
        lips_points.append((x, y))

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array(lips_points)], color)

    # Blend lipstick
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

# Skin tone detection (basic ML logic)
def detect_skin_tone(image):
    avg_color = np.mean(image.reshape(-1, 3), axis=0)

    if avg_color[2] > 150:
        return "Warm"
    elif avg_color[1] > 120:
        return "Neutral"
    else:
        return "Cool"

# Recommend lipstick
def recommend_shades(tone):
    if tone == "Warm":
        return ["Red ❤️", "Nude 🤎"]
    elif tone == "Cool":
        return ["Pink 💗", "Plum 🍇"]
    else:
        return ["Pink 💗", "Nude 🤎"]

# Upload image
uploaded_file = st.file_uploader("📤 Upload Your Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Detect skin tone
        tone = detect_skin_tone(img)
        st.success(f"🧠 Detected Skin Tone: {tone}")

        recommended = recommend_shades(tone)
        st.info(f"💡 Recommended Shades: {', '.join(recommended)}")

        mode = st.radio("🎨 Select Mode", ["Single Shade", "Compare Shades"])

        if mode == "Single Shade":
            choice = st.selectbox("💄 Choose Lipstick", list(lip_colors.keys()))

            output = apply_lipstick(img.copy(), landmarks, lip_colors[choice])
            st.image(output, caption="✨ Final Look", use_column_width=True)

        else:
            st.subheader("🔍 Compare Shades")

            cols = st.columns(2)
            color_list = list(lip_colors.keys())

            for i in range(4):
                output = apply_lipstick(img.copy(), landmarks, lip_colors[color_list[i]])
                cols[i % 2].image(output, caption=color_list[i], use_column_width=True)

    else:
        st.error("❌ No face detected! Please upload a clear image.")
