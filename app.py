import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Title
st.title("🎨 Sketch to Image Converter")

st.write("Draw a sketch and convert it into a stylized image!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=5,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Function to convert sketch → image
def process_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Invert edges
    edges_inv = cv2.bitwise_not(edges)

    # Apply color map (fake "realistic" effect)
    colored = cv2.applyColorMap(edges_inv, cv2.COLORMAP_JET)

    return edges, colored

# When user draws something
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)

    # Convert RGBA → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    edges, result = process_image(img)

    st.subheader("🖼️ Output")

    col1, col2 = st.columns(2)

    with col1:
        st.image(edges, caption="Sketch (Edges)", use_column_width=True)

    with col2:
        st.image(result, caption="Converted Image", use_column_width=True)
