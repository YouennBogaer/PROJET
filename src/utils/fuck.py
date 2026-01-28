import streamlit as st
from PIL import Image
from pathlib import Path
from Model import Model
import os
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="Vision Chatbot")

st.title("Q/A image ")
st.markdown("Dl une image")

# --- INITIALISATION DU MODÈLE ---
# On définit un prompt par défaut, mais l'utilisateur pourra le changer via le chat
DEFAULT_PROMPT = "Describe this image in detail."


uploaded_file = st.file_uploader("Choisis une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)


    temp_path = "999999.jpg"
    image.save(temp_path)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Image téléchargée", use_container_width=True)

    # --- ZONE DE CHAT ---
    with col2:
        st.subheader("Conversation")


        if "messages" not in st.session_state:
            st.session_state.messages = []


        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        if user_query := st.chat_input("Dis-moi quoi faire avec cette image..."):

            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)


            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    try:
                        # On passe une liste vide ou factice pour coco_captions car on n'en a pas besoin ici
                        chat_model = Model(
                            model_name=os.getenv("DEFAULT_MODEL",""),
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
                        st.error(f"Erreur lors de la génération : {e}")
                        print(f"error !!!!!!!!!!!! :{e}")

else:
    st.info("Veuillez uploader une image pour commencer à discuter.")

# Delete historique de msg
if st.sidebar.button("Effacer l'historique"):
    st.session_state.messages = []
    st.rerun()