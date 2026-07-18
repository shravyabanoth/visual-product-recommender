import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

from src.feature_extraction import build_feature_extractor
from src.similarity_search import SimilaritySearch


st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="🛍️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

SIAMESE_MODEL_PATH = "models/siamese_embedding_model.keras"
TRANSFER_MODEL_PATH = "models/transfer_learning_model.keras"


def load_transfer_feature_model():
    """
    Loads the fine-tuned transfer-learning classifier and strips off the
    classification head, exposing the second-to-last dense layer as the
    embedding output (mirrors src/generate_transfer_embeddings.py).
    """
    model = tf.keras.models.load_model(TRANSFER_MODEL_PATH, compile=False)
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output,
    )
    return feature_model, tf.keras.applications.resnet50.preprocess_input


# Each entry: display name -> (embeddings file, function that returns (model, preprocess_fn))
MODEL_CONFIGS = {
    "Baseline (ResNet50)": {
        "embeddings": "embeddings/baseline_embeddings.npz",
        "loader": lambda: build_feature_extractor("resnet50"),
    },
    "Transfer Learning": {
        "embeddings": "embeddings/transfer_embeddings.npz",
        "loader": load_transfer_feature_model,
    },
    "Siamese Network": {
        "embeddings": "embeddings/siamese_embeddings.npz",
        "loader": lambda: build_feature_extractor("resnet50", weights_path=SIAMESE_MODEL_PATH),
    },
}


@st.cache_resource(show_spinner="Loading embeddings index...")
def load_search_engine(embeddings_path: str) -> SimilaritySearch:
    return SimilaritySearch(embeddings_path)


@st.cache_resource(show_spinner="Loading feature extractor model...")
def load_feature_model(model_name: str):
    return MODEL_CONFIGS[model_name]["loader"]()


def get_available_models() -> dict:
    """Only offer models whose embeddings file actually exists on disk."""
    return {
        name: cfg
        for name, cfg in MODEL_CONFIGS.items()
        if os.path.exists(cfg["embeddings"])
    }


def embed_uploaded_image(image: Image.Image, preprocess_fn, model) -> np.ndarray:
    """Runs the uploaded PIL image through the feature extractor directly
    in memory (no JPEG re-encode/reload), so preprocessing matches the
    catalog embedding pipeline exactly (src/feature_extraction.py)."""
    img = image.resize((224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_fn(arr)

    embedding = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    return embedding


def render_recommendations(recommendations: list):
    cols = st.columns(len(recommendations)) if recommendations else []
    for col, item in zip(cols, recommendations):
        with col:
            if os.path.exists(item["filepath"]):
                st.image(item["filepath"], use_container_width=True)
            else:
                st.warning("Image file not found on disk.")
            st.write(f"**ID:** {item['id']}")
            st.write(f"**Similarity:** {item['similarity']:.3f}")
            st.write(f"**Category:** {item['category']}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🛍️ Visual Product Recommendation System")

st.markdown(
    "Upload a fashion product image and the system will retrieve the most "
    "visually similar products using deep learning."
)

available_models = get_available_models()

if not available_models:
    st.error(
        "No embeddings found. Run the feature extraction / embedding "
        "generation scripts first (see src/feature_extraction.py, "
        "src/generate_transfer_embeddings.py, src/generate_siamese_embeddings.py)."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def get_supported_categories(embeddings_path: str) -> list:
    data = np.load(embeddings_path, allow_pickle=True)
    return sorted(set(data["labels"].tolist()))


# All models share the same underlying subset, so any one embeddings file
# tells us the full category coverage of this demo.
_first_embeddings_path = next(iter(available_models.values()))["embeddings"]
supported_categories = get_supported_categories(_first_embeddings_path)

with st.container(border=True):
    st.markdown("**📦 Supported Product Categories**")
    st.caption(
        "This system is trained and indexed on a curated subset of the "
        "Fashion Product Images dataset. For best results, upload an "
        "image from one of the categories below."
    )
    badges = " ".join(
        f'<span style="background-color:#262730;border:1px solid #4a4a4a;'
        f'border-radius:14px;padding:4px 14px;margin:3px;display:inline-block;'
        f'font-size:13px;color:#fafafa;">{cat}</span>'
        for cat in supported_categories
    )
    st.markdown(badges, unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.header("Settings")

    compare_mode = st.checkbox(
        "Compare all available models side by side",
        value=False,
        help="Runs the same uploaded image through every model you've "
        "generated embeddings for, so you can visually compare results.",
    )

    if not compare_mode:
        selected_model_name = st.selectbox(
            "Model",
            list(available_models.keys()),
        )

    k = st.slider("Top K Recommendations", 1, 10, 5)

uploaded = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=250)
    st.divider()

    if compare_mode:
        tabs = st.tabs(list(available_models.keys()))

        for tab, model_name in zip(tabs, available_models.keys()):
            with tab:
                with st.spinner(f"Running {model_name}..."):
                    model, preprocess_fn = load_feature_model(model_name)
                    search_engine = load_search_engine(
                        available_models[model_name]["embeddings"]
                    )
                    embedding = embed_uploaded_image(image, preprocess_fn, model)
                    recommendations = search_engine.query(embedding, k=k)

                st.subheader(f"Recommended Products — {model_name}")
                render_recommendations(recommendations)

    else:
        with st.spinner(f"Running {selected_model_name}..."):
            model, preprocess_fn = load_feature_model(selected_model_name)
            search_engine = load_search_engine(
                available_models[selected_model_name]["embeddings"]
            )
            embedding = embed_uploaded_image(image, preprocess_fn, model)
            recommendations = search_engine.query(embedding, k=k)

        st.subheader(f"Recommended Products — {selected_model_name}")
        render_recommendations(recommendations)