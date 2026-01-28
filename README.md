# Projet NLP pour 19/12/2025
## Sujet : Multimodal

# Tâche : Image Captioning
# Choix du llm : LLaVA (via ollama)
# Choix de baseline : 

# src/utils/dl.py : télécharger COCO2017 val avec captions (~2GB)

## JAVA 8 : pour les métriques METEOR et SPICE

# il faut téléchargé moondream et qwen2.5vl:3b en faisant ollama pull [nom du modèle]

j'ai créé un .env dans celui la on met les noms des différentes variable (java, dossier coco) !! pour windows c'est java.exe qu'il faut mettre et linux juste java
# GPU
Pour les gpu peut etre que la vram va pas se décharger en testant tout les modèle d'affiler il faudrait peut-être rajouter qqchose je vais regarder demain