
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from datetime import datetime

# --------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Vision | Defect Detection",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hazelnut_autoencoder.keras"

THRESHOLD = 0.00029319
PIXEL_THRESHOLD = 0.002

# --------------------------------
# 2. PROFESSIONAL DASHBOARD STYLE
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #172337;
}

h1, h2, h3 {
    color: #E5EFFF;
}

div[data-testid="stMetric"] {
    background-color: #1C2B40;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #33445C;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --------------------------------
# 4. HEADER
# --------------------------------

st.title("🔍 AI VISION")
st.subheader("Intelligent Industrial Defect Detection System")

st.caption(
    "CNN-based visual inspection | Anomaly detection | "
    "Automated quality control"
)

st.divider()

# --------------------------------
# 5. SIDEBAR
# --------------------------------

st.sidebar.title("CONTROL PANEL")

product = st.sidebar.selectbox(
    "Select Product Category",
    [
        "Hazelnut",
        "Bottle (Coming Soon)",
        "Capsule (Coming Soon)",
        "Carpet (Coming Soon)",
        "Grid (Coming Soon)",
        "Leather (Coming Soon)"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model Information")
st.sidebar.write("Architecture: CNN Autoencoder")
st.sidebar.write("Input size: 128 × 128")
st.sidebar.write("Channels: RGB")
st.sidebar.write("Current model: Hazelnut")

st.sidebar.markdown("---")

st.sidebar.info(
    "Currently, only the Hazelnut category has a trained model."
)

# --------------------------------
# 6. SESSION HISTORY
# --------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# 7. CHECK PRODUCT CATEGORY
# --------------------------------

if product != "Hazelnut":
    st.warning(
        "This category is planned for future expansion. "
        "A separately trained model is required before inspection."
    )
    st.stop()

# --------------------------------
# 8. LOAD MODEL SAFELY
# --------------------------------

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found! "
        "Place hazelnut_autoencoder.keras in the main project folder."
    )
    st.stop()

try:
    model = load_model()
    st.sidebar.success("AI Model Loaded Successfully")

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# --------------------------------
# 9. IMAGE UPLOAD
# --------------------------------

st.markdown("## Product Inspection")

uploaded_file = st.file_uploader(
    "Upload a hazelnut product image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Product Image",
            use_container_width=True
        )

    with col2:
        st.markdown("### Inspection Details")
        st.write("Product: Hazelnut")
        st.write("Image: " + uploaded_file.name)
        st.write("Model: CNN Autoencoder")

        inspect = st.button("RUN AI INSPECTION")

    # --------------------------------
    # 10. PREDICTION
    # --------------------------------

    if inspect:

        with st.spinner("AI model is inspecting the product..."):

            # Preprocessing
            processed = image.resize((128, 128))

            image_array = np.asarray(
                processed,
                dtype=np.float32
            ) / 255.0

            original = np.expand_dims(
                image_array,
                axis=0
            )

            # Model reconstruction
            reconstructed = model.predict(
                original,
                verbose=0
            )

            # Calculate anomaly map
            error_map = np.mean(
                np.square(original - reconstructed),
                axis=-1
            )[0]

            # Overall reconstruction error
            error = float(np.mean(error_map))

            # Classification
            if error > THRESHOLD:
                result = "DEFECTIVE"
            else:
                result = "NORMAL"

            # Binary defect mask
            defect_mask = (
                error_map > PIXEL_THRESHOLD
            ).astype(np.uint8)

            highlighted_pixels = int(
                np.sum(defect_mask)
            )

            # Store result
            record = {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Product": "Hazelnut",
                "Image": uploaded_file.name,
                "Result": result,
                "Error": error
            }

            st.session_state.history.append(record)

        # --------------------------------
        # 11. RESULT DISPLAY
        # --------------------------------

        st.divider()
        st.markdown("## Inspection Results")

        if result == "DEFECTIVE":

            st.error("⚠️ DEFECTIVE PRODUCT DETECTED")

        else:

            st.success("✅ NORMAL PRODUCT")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Reconstruction Error",
            f"{error:.7f}"
        )

        m2.metric(
            "Classification",
            result
        )

        m3.metric(
            "Highlighted Pixels",
            f"{highlighted_pixels:,}"
        )

        # --------------------------------
        # 12. VISUAL ANALYSIS
        # --------------------------------

        st.markdown("## Visual Defect Analysis")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image_array)
        axes[0].set_title("Original Image")

        axes[1].imshow(
            np.clip(reconstructed[0], 0, 1)
        )
        axes[1].set_title("Reconstructed Image")

        axes[2].imshow(error_map, cmap="jet")
        axes[2].set_title("Anomaly Heatmap")

        axes[3].imshow(defect_mask, cmap="gray")
        axes[3].set_title("Binary Defect Mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------
        # 13. INSPECTION INFORMATION
        # --------------------------------

        st.markdown("## Inspection Summary")

        st.write(f"**Product:** Hazelnut")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Reconstruction Error:** {error:.8f}")
        st.write(f"**Classification Threshold:** {THRESHOLD}")
        st.write(
            "**Highlighted Pixel Count:** "
            f"{highlighted_pixels}"
        )

        st.caption(
            "Experimental prototype: classification and "
            "localization thresholds require validation "
            "against a representative dataset before "
            "industrial use."
        )

# --------------------------------
# 14. INSPECTION HISTORY
# --------------------------------

st.divider()

st.markdown("## Recent Inspection History")

if st.session_state.history:

    st.dataframe(
        st.session_state.history[::-1],
        use_container_width=True
    )

    if st.button("Clear Inspection History"):
        st.session_state.history = []
        st.rerun()

else:

    st.info("No inspections performed yet.")

# --------------------------------
# 15. FOOTER
# --------------------------------

st.divider()

st.caption(
    "AI VISION | CNN-Based Industrial Quality Inspection "
    "| Academic Project Prototype"
)
