# Projet NLP pour 19/12/2025
## Sujet : Multimodal

# Tâche : Image Captioning
# Choix du llm : LLaVA (via ollama)
# Choix de baseline : 

# Image Captioning & RAG Pipeline

🚀 **Description** : Système de génération de légendes d'images optimisé par RAG (Retrieval Augmented Generation) pour améliorer la pertinence contextuelle et limiter les hallucinations visuelles.

---

## 📂 Structure du Projet

.
├── app.py                      # Application principale (Streamlit)
├── main.ipynb                  # Expérimentations et développement
├── main_compare.ipynb          # Benchmarks et comparaisons de modèles
├── core/                       # Logique métier
│   ├── Model.py                # Architecture et inférence du modèle
│   └── rag.py                  # Moteur de recherche et contexte RAG
├── data/                       # Données et ressources
│   ├── our_data/               # Index (captions_map) et synonymes SOTA
│   └── test/                   # Images de test (dog, food, etc.)
├── evaluation/                 # Métriques de performance
│   ├── ChairScorer.py          # Analyse des hallucinations (CHAIR)
│   ├── MeteorScorer.py         # Score METEOR
│   └── Scorer.py               # Orchestrateur d'évaluation
├── utils/                      # Scripts utilitaires
│   ├── dictCaptions.py         # Helpers pour dictionnaires de légendes
│   └── dl.py                   # Téléchargement de modèles/assets
└── temp_rag_images/            # Traitements temporaires pour le RAG

---

## ⚙️ Installation & Configuration

### 1. Environnement (Conda)
conda create -n nlp_proj python=3.13
conda activate nlp_proj
pip install -r requirements.txt

### 2. Installation et Téléchargement
Utilisez le fichier requirements.txt pour installer toutes les bibliothèques nécessaires :
pip install -r requirements.txt


## 🛠️ Configuration du Système & Dépendances

### 1. Installation d'Ollama et des Modèles
Ollama est requis pour faire tourner les modèles de vision localement. Installez-le puis récupérez les modèles nécessaires :

```bash
# Installation d'Ollama (Linux)
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Téléchargement des modèles de vision
ollama pull llava
ollama pull moondream
ollama pull qwen2.5-vl:3b
```

# Installation de SDKMAN
```bash
curl -s "[https://get.sdkman.io](https://get.sdkman.io)" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
```
# Installation de Java 8
```bash
sdk install java 8.0.402-amzn
```

# --- CONFIGURATION LOCALE ---

# Chemin absolu vers votre dataset COCO
DATASET_PATH="/mnt/2210B8B210B88E73/Desktop/IA_Image/coco2017/"

# Modèles disponibles via Ollama
AVAILABLE_MODELS="llava,moondream,qwen2.5-vl:3b"
DEFAULT_MODEL="llava"

# Chemin vers votre exécutable Java 8 (Exemple avec SDKMAN)
JAVA_PATH="/home/vic/.sdkman/candidates/java/current/bin/java"
---

## 🚀 Utilisation

### Lancer l'interface utilisateur
streamlit run app.py

### Exécuter les analyses
Ouvrez main.ipynb pour tester le pipeline complet ou main_compare.ipynb pour visualiser les différences de performances entre les configurations.

---

## 📊 Évaluation
Le projet intègre des métriques spécifiques au NLP et à la Vision :
* CHAIR : Mesure le taux d'objets hallucinés non présents dans l'image.
* METEOR : Évalue la qualité grammaticale et sémantique.

---

## 🛠️ Stack Technique
* Langage : Python 3.13
* Interface : Streamlit
* Analyse : Notebooks Jupyter
* Ressources : COCO Synonyms, Captions Map

---

## ✍️ Auteur
* Vic