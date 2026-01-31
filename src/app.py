import streamlit as st
from PIL import Image
from core.Model import Model
import os
import shutil
from dotenv import load_dotenv
from core.rag import pipeline_model, pipeline_clip
from pathlib import Path
load_dotenv()


SRC_DIR = Path(__file__).parent.parent
TIGER_PATH = SRC_DIR / "assets" / "tiger.gif"
AUDIO_PATH = SRC_DIR / "assets" / "dame-un-gr.mp3"

#CONFIGURATION
st.set_page_config(page_title="Vision Chatbot", layout="wide")

# Load model options from environment variables
env_models = os.getenv("AVAILABLE_MODELS", "llava")
available_models = [m.strip() for m in env_models.split(",")]
default_model_env = os.getenv("DEFAULT_MODEL", "llava")

TEMP_DIR = "temp_rag_images"


def save_uploaded_files(uploaded_files):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(os.path.abspath(file_path))
    return saved_paths


#SIDEBAR

st.sidebar.title("Settings")
try:
    default_index = available_models.index(default_model_env)
except ValueError:
    default_index = 0

selected_model = st.sidebar.selectbox(
    "Select AI Model:",
    available_models,
    index=default_index
)
st.sidebar.info(f"Active Model: {selected_model}")

mode = st.sidebar.radio("Analysis Mode:", ["Single Image Analysis", "Multimodal RAG"])

st.sidebar.markdown("---")
if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    st.rerun()


st.sidebar.markdown("<br>" * 25, unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns([3, 2, 3])
with col2:
    if st.button(":tiger2:", help="Un qué ?"):
        st.sidebar.image(TIGER_PATH, use_container_width=True)
        audio_file = open(AUDIO_PATH, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3',autoplay=True)
        st.balloons()



# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MODE 1: SINGLE IMAGE ANALYSIS ---
if mode == "Single Image Analysis":
    st.title(f"Image Analysis with {selected_model}")
    st.markdown("Ask a question about a specific image.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        temp_path = "temp_single.jpg"
        image.save(temp_path)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Chat")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Ask a question..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner(f"Analyzing with {selected_model}..."):
                        try:
                            chat_model = Model(
                                model_name=selected_model,
                                prompts=[user_query],
                                imgs_path=[temp_path],
                                coco_captions={}
                            )
                            responses, _ = chat_model.execute(prompt_id=0, freq_print=0)

                            img_key = list(responses.keys())[0]
                            bot_response = responses[img_key][0]

                            st.markdown(bot_response)
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})

                        except Exception as e:
                            st.error(f"Error: {e}")

#MODE 2: MULTIMODAL RAG
else:
    st.title("Multimodal RAG")
    st.markdown(f"Search using CLIP and analyze with {selected_model}")

    uploaded_files = st.file_uploader("Upload photo album", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if len(uploaded_files) > 1:
        image_paths = save_uploaded_files(uploaded_files)
        st.success(f"{len(image_paths)} images loaded.")

        with st.expander("View Gallery"):
            cols = st.columns(5)
            for idx, img_file in enumerate(uploaded_files):
                cols[idx % 5].image(img_file, caption=img_file.name, use_container_width=True)

        if user_query := st.chat_input("Search for something"):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Processing RAG pipeline..."):
                    try:
                        best_img_path = pipeline_clip(user_query, image_paths)

                        if best_img_path is None:
                            st.error("Clip did not find a good match, please retry with another prompt")
                        else:
                            st.markdown(f"**Best match:** {os.path.basename(best_img_path)}")
                            st.image(best_img_path, width=300)
                            bot_response = pipeline_model(user_query, best_img_path, model=selected_model)
                            st.markdown("### Answer")
                            st.markdown(bot_response)

                    except Exception as e:
                        st.error(f"RAG Pipeline Error: {e}")
    else:
        st.info("Please upload at least 2 images to enable RAG mode.")
