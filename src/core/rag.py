from dotenv import load_dotenv
import torch
from pathlib import Path
import re

load_dotenv()
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import time
import os
from core.Model import Model
import gc

# load
# CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# load clip model
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def analis_clip(questions, images_path, treshold=0.4):
    """
    Une fonction qui applqiue le modèle de clip (d'open-ai)et renvoies  la photo qui a le plus de chance de correspondre au prompt
    :param questions: prompt
    :param images_path:
    :param treshold: le seuil de confiance
    :return:
    """
    images = [Image.open(path) for path in images_path]
    if len(images) <= 1 or questions == "":
        print("erreur il ya moins d'une image ou la question est vide")
    else:
        inputs = processor_clip(text=questions, return_tensors="pt", images=images, padding=True).to(device)

        # probs calcul
        outputs = model_clip(**inputs)
        probs = outputs.logits_per_image.softmax(dim=0)

        best_idx = probs.argmax().item()
        best_score = probs.max().item()

        # si on est en dessous du seuil de confiance alors on ne renvoie rien
        if best_score > treshold:

            print(f"Image choisie : {images_path[best_idx]} (Confiance: {best_score:.2f})")
            return images_path[best_idx]
        else:
            print("Pas de bonne image trouvé")
            return None


def pipeline_rag(questions, images_path, model=os.getenv("DEFAULT_MODEL")):
    """
    Une fonction qui devait faire toute la pipeline du RAG mais ensuite on a découpé en deux pour pouvoir
    affciher sur l'app
    :param questions:
    :param images_path:
    :param model:
    :return:
    """
    prompt_clip = preprocess_prompt(questions)
    best_img = analis_clip(prompt_clip, images_path)

    if best_img is None:
        return None, None

    def run_inf(p):
        chat_model = Model(
            model_name=model,
            prompts=[questions],
            imgs_path=[p],
            coco_captions={}
        )
        res, _ = chat_model.execute(prompt_id=0, freq_print=0)
        return res[list(res.keys())[0]][0]

    try:
        print("Try 1...")
        bot_response = run_inf(best_img)
    except Exception as e:
        print(f"Err 1: {e}")
        try:
            print("Resize...")
            with Image.open(best_img) as img:
                # Force multiples de 28 (pour le décupage en patch)
                new_img = img.resize((448, 448), Image.LANCZOS)
                tmp = best_img.replace(".", "_tmp.")
                new_img.save(tmp)
                best_img = tmp

            bot_response = run_inf(best_img)
            print("Ok 2")
        except Exception as e2:
            print(f"Err 2: {e2}")
            return None, best_img

    gc.collect()
    print(f"response : {bot_response}")
    return bot_response, best_img


def pipeline_clip(questions, images_path):
    """
    Pipeline complète qui utilise le modèle clip
    :param questions:
    :param images_path:
    :return:
    """
    prompt_clip = preprocess_prompt(questions)
    best_img = analis_clip(prompt_clip, images_path)
    if best_img is None:
        return None
    return best_img


def pipeline_model(questions, image, model=os.getenv("DEFAULT_MODEL")):
    """
    pipeline complète pour le modèle ollama
    :param questions:
    :param image:
    :param model:
    :return:
    """

    def run_inf(p):
        chat_model = Model(
            model_name=model,
            prompts=[questions],
            imgs_path=[p],
            coco_captions={}
        )
        res, _ = chat_model.execute(prompt_id=0, freq_print=0)
        return res[list(res.keys())[0]][0]

    # La on fait un try si l'image passe pas on redimenssionne puis on retente (c'est à cause de certains formats
    # d'image (.webpng qui peuvent être mal encoé mais qu   nd m^mee s'appeler .png)
    try:
        print("Try 1...")
        bot_response = run_inf(image)
    except Exception as e:
        print(f"Err 1: {e}")
        try:
            print("Resize...")
            with Image.open(image) as img:
                # Force multiples de 28 (pour le décupage en patch)
                new_img = img.resize((448, 448), Image.LANCZOS)
                tmp = image.replace(".", "_tmp.")
                new_img.save(tmp)
                best_img = tmp

            bot_response = run_inf(best_img)
            print("Ok 2")
        except Exception as e2:
            print(f"Err 2: {e2}")
            return "Ann error occured on the image size or type", best_img
    print(f"response : {bot_response}")
    return bot_response


def preprocess_prompt(prompt):
    """
    preprocess prompt for CLIP on met en minuscule et on enlève les caractères spéciaux

    :param texte:
    :return:
    """
    prompt = prompt.lower().strip()
    clean_prompt = re.sub(r'[^a-zA-Z0-9\s?.,!]', '', prompt)
    clean_prompt = " ".join(clean_prompt.split())

    final_prompt = f"a photo of {clean_prompt}"
    return final_prompt


if __name__ == "__main__":
    if __name__ == "__main__":
        ROOT = Path(__file__).resolve().parent.parent
        test_dir = ROOT / "data" / "test"

        image_names = ["bathroom.png", "garage.png", "food.png", "dog.png", "dogs.png"]
        image_paths = [str(test_dir / img) for img in image_names if (test_dir / img).exists()]

        if not image_paths:
            print(f"Erreur : Aucun fichier trouvé dans {test_dir}")
        else:
            user_query = "Une facture de téléphone"
            target_model = "moondream"

            start_time = time.time()

            selected_image = pipeline_clip(user_query, image_paths)

            if selected_image:
                response = pipeline_model(user_query, selected_image, model=target_model)

                print(f"\nPrompt: {user_query}")
                print(f"Image: {Path(selected_image).name}")
                print("-" * 10)
                print(f"Réponse: {response}")
            else:
                print("Aucune image correspondante trouvée.")

            print(f"\nDurée: {time.time() - start_time:.2f}s")
