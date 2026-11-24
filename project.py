
import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw
from io import BytesIO

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Shape Detection Studio",
    page_icon="🔷",
    layout="wide"
)

# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10172a, #182443, #10172a);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg,#00e5ff,#a855f7,#ff4ecd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1 !important;
    font-size: 17px;
}

div.stButton > button {
    background: linear-gradient(90deg,#635bff,#a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    padding: 10px;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg,#ec4899,#6366f1);
    transform: scale(1.03);
}

[data-testid="stMetric"] {
    background: #202d49;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #465575;
}

section[data-testid="stFileUploader"] {
    background: #202d49;
    border-radius: 15px;
    padding: 15px;
}

hr {
    border-color: #465575;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🔷 SHAPE DETECTION STUDIO</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">An Interactive Computer Vision Experience</p>',
    unsafe_allow_html=True
)

st.write("")

# ==========================================
# SHAPE DRAWING FUNCTION
# ==========================================

def get_shape_points(name, cx, cy, radius):

    if name == "Triangle":
        angles = [-90, 30, 150]
    elif name == "Square":
        angles = [-45, 45, 135, 225]
    elif name == "Pentagon":
        angles = [-90, -18, 54, 126, 198]
    elif name == "Hexagon":
        angles = [i * 60 - 90 for i in range(6)]
    elif name == "Octagon":
        angles = [i * 45 - 90 for i in range(8)]
    elif name == "Star":
        points = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.42
            angle = math.radians(i * 36 - 90)
            points.append((
                int(cx + r * math.cos(angle)),
                int(cy + r * math.sin(angle))
            ))
        return points
    elif name == "Diamond":
        angles = [-90, 0, 90, 180]
        radius_x, radius_y = radius * 0.7, radius
        return [
            (int(cx + radius_x * math.cos(math.radians(a))),
             int(cy + radius_y * math.sin(math.radians(a))))
            for a in angles
        ]
    elif name == "Rhombus":
        return [
            (cx, cy-radius),
            (cx+radius, cy-radius//2),
            (cx, cy+radius),
            (cx-radius, cy+radius//2)
        ]
    else:
        angles = [i * 360 / 7 - 90 for i in range(7)]

    return [
        (
            int(cx + radius * math.cos(math.radians(a))),
            int(cy + radius * math.sin(math.radians(a)))
        )
        for a in angles
    ]


def create_shape_image(name, color):

    image = Image.new("RGB", (220, 160), "#202d49")
    draw = ImageDraw.Draw(image)

    cx, cy = 110, 80
    radius = 48

    if name == "Circle":
        draw.ellipse(
            (cx-radius, cy-radius, cx+radius, cy+radius),
            fill=color
        )

    elif name == "Oval":
        draw.ellipse(
            (cx-65, cy-38, cx+65, cy+38),
            fill=color
        )

    elif name == "Rectangle":
        draw.rectangle(
            (cx-65, cy-40, cx+65, cy+40),
            fill=color
        )

    else:
        points = get_shape_points(name, cx, cy, radius)
        draw.polygon(points, fill=color)

    return image


# ==========================================
# SHAPE GALLERY
# ==========================================

st.header("🎨 Explore the Shape Gallery")
st.write("Choose a sample shape from the collection.")

shapes = [
    "Triangle",
    "Square",
    "Rectangle",
    "Circle",
    "Oval",
    "Pentagon",
    "Hexagon",
    "Octagon",
    "Star",
    "Diamond",
    "Rhombus",
    "Heptagon"
]

colors = [
    "#00D9FF",
    "#A855F7",
    "#FF4ECD",
    "#FFD43B",
    "#32D583",
    "#FF7043",
    "#60A5FA",
    "#F472B6",
    "#FACC15",
    "#FB7185",
    "#34D399",
    "#A78BFA"
]

if "selected_shape" not in st.session_state:
    st.session_state.selected_shape = None

for start in range(0, len(shapes), 4):

    cols = st.columns(4)

    for j, col in enumerate(cols):

        index = start + j

        if index >= len(shapes):
            break

        name = shapes[index]

        with col:

            st.image(
                create_shape_image(name, colors[index]),
                use_container_width=True
            )

            if st.button(
                "Select Shape",
                key=f"shape_{name}",
                use_container_width=True
            ):
                st.session_state.selected_shape = name

if st.session_state.selected_shape:

    st.success(
        f"Selected shape: {st.session_state.selected_shape}"
    )

    st.balloons()

    if st.button("Clear Selection"):
        st.session_state.selected_shape = None
        st.rerun()

st.divider()

# ==========================================
# IMAGE UPLOAD SECTION
# ==========================================

st.header("📸 Upload Image for Detection")

uploaded_file = st.file_uploader(
    "Upload an image containing geometric shapes",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# SHAPE CLASSIFICATION
# ==========================================

def classify_shape(contour):

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Other", None

    approx = cv2.approxPolyDP(
        contour,
        0.018 * perimeter,
        True
    )

    corners = len(approx)

    area = cv2.contourArea(contour)

    if area == 0:
        return "Other", approx

    x, y, w, h = cv2.boundingRect(approx)
    ratio = w / float(h)

    circularity = (
        4 * np.pi * area / (perimeter * perimeter)
    )

    # Circle and oval
    if circularity > 0.82:
        if 0.85 <= ratio <= 1.15:
            return "Circle", approx
        return "Oval", approx

    # Triangle
    if corners == 3:
        return "Triangle", approx

    # Four-sided shapes
    if corners == 4:

        pts = approx.reshape(4, 2).astype(float)

        sides = [
            np.linalg.norm(pts[i] - pts[(i+1) % 4])
            for i in range(4)
        ]

        side_ratio = max(sides) / max(min(sides), 1)

        if 0.85 <= ratio <= 1.15 and side_ratio < 1.2:
            return "Square", approx

        if side_ratio < 1.2:
            return "Rhombus", approx

        if 0.8 <= ratio <= 1.2:
            return "Diamond", approx

        return "Rectangle", approx

    # Other polygons
    if corners == 5:
        return "Pentagon", approx

    if corners == 6:
        return "Hexagon", approx

    if corners == 7:
        return "Heptagon", approx

    if corners == 8:
        return "Octagon", approx

    if corners == 10:
        return "Star", approx

    return "Other", approx


# ==========================================
# IMAGE PROCESSING
# ==========================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:

        st.error("Unable to read the uploaded image.")

    else:

        st.subheader("🖼️ Original Image")

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()

        counts = {shape: 0 for shape in shapes}
        counts["Other"] = 0

        detection_colors = [
            (0, 255, 255),
            (255, 0, 255),
            (0, 255, 0),
            (255, 150, 0),
            (0, 150, 255),
            (255, 255, 0)
        ]

        min_area = st.slider(
            "Minimum object area",
            min_value=100,
            max_value=3000,
            value=300,
            step=100
        )

        # ==================================
        # DETECTION
        # ==================================

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            shape, approx = classify_shape(contour)

            if shape == "Other":
                counts["Other"] += 1
            else:
                counts[shape] += 1

            x, y, w, h = cv2.boundingRect(contour)

            color = detection_colors[
                sum(counts.values()) % len(detection_colors)
            ]

            cv2.drawContours(
                result,
                [approx],
                -1,
                color,
                3
            )

            cv2.putText(
                result,
                shape,
                (x, max(y-10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # ==================================
        # DISPLAY RESULTS
        # ==================================

        st.divider()

        st.header("🔍 Detection Results")

        tab1, tab2, tab3 = st.tabs([
            "✨ Final Detection",
            "⚡ Edge Detection",
            "📊 Statistics"
        ])

        with tab1:

            st.image(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            success, encoded = cv2.imencode(
                ".png", result
            )

            if success:
                st.download_button(
                    "⬇️ Download Detection Image",
                    data=encoded.tobytes(),
                    file_name="shape_detection.png",
                    mime="image/png"
                )

        with tab2:

            st.image(
                edges,
                caption="Detected Edges",
                use_container_width=True
            )

        with tab3:

            total = sum(counts.values())

            c1, c2 = st.columns(2)

            c1.metric("Total Shapes", total)

            c2.metric(
                "Different Shape Types",
                sum(1 for value in counts.values() if value > 0)
            )

            st.subheader("Individual Shape Counts")

            for shape, count in counts.items():

                if count > 0:

                    st.write(f"**{shape}**")

                    st.progress(
                        count / max(total, 1)
                    )

                    st.caption(f"Detected: {count}")

            if total == 0:
                st.warning("No shapes detected. Try another image.")

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
    <h3>🔷 Shape Detection Studio</h3>
    <p>Powered by Python | OpenCV | NumPy | Streamlit</p>
    <p>Computer Vision Mini Project</p>
    </center>
    """,
    unsafe_allow_html=True
)

