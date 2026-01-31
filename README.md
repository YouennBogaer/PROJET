
# Projet NLP Multimodal : Description d'Images & VQA Local

**Ewen DANO, Victor ANDRE, Youenn BOGAER**

**Date de rendu :** 31/01/2025

**Sujet :** NLP Multimodal (Sujet 03)

**Tâche :** Image Captioning & Visual Question Answering (VQA)

**Technique avancée :** RAG Multimodal & Fusion Multimodale (Late Fusion)

---

## 1. Présentation du Projet

Ce projet implémente une solution de compréhension d'image évolutive, passant d'un système de **Captioning** (génération de descriptions) à un outil interactif de **VQA** (Visual Question Answering) via Streamlit. L'innovation repose sur un moteur de **RAG Multimodal** capable de naviguer intelligemment dans une base de données visuelle.

Trouvez une démonstration en vidéo : https://youtu.be/G_b3ViNHbyA

---

## 2. Architecture Technique & Fusion Multimodale

Le projet repose sur une **fusion multimodale de type "Late Fusion"**, combinant la recherche vectorielle et le raisonnement génératif.

### 2.1 RAG 

Pour l'extension RAG, nous utilisons **CLIP** pour projeter texte et images dans un espace vectoriel commun. Le processus de sélection est optimisé comme suit :

1. **Calcul de Similarité** : Nous calculons la probabilité de correspondance entre la requête textuelle et chaque image de la base.
2. **Filtrage de Confiance** : Nous sélectionnons l'image ayant la **probabilité maximale**.
3. **Gestion de l'Incertitude** : Si le score de probabilité est **trop bas (en dessous d'un seuil défini)**, le système considère qu'aucune image n'est pertinente. Cela évite au LLM de générer une réponse basée sur un contexte visuel erroné (limitation des hallucinations "hors contexte").

### 2.2 Stratégie d'Analyse (Prompt Engineering & Modèles)

Nous n'avons pas testé une solution unique, mais une matrice de combinaisons pour trouver la meilleure performance :

* **Comparaison de Modèles** : Benchmarks entre `Llava`, `Moondream` et `Qwen2.5-VL` (via Ollama).
* **Prompt Engineering** : Test de **différents prompts** (plus ou moins descriptifs, avec ou sans pré-prompts de contexte) pour observer l'impact sur la précision des réponses et le taux d'hallucination.

---

## 3. Démarche : De la Description au VQA

* **Baseline (Tronc commun)** : Génération de légendes simples (Captioning) sur une image isolée fournie par l'utilisateur, nous avons commencé avec llava puis tester des modèles plus légers (monndream...) pour comparer..
* **Extension Avancée** : Transformation de l'outil en système de VQA dynamique. Grâce au RAG, l'utilisateur pose une question et le système retrouve l'image source avant de répondre.

---

## 4. Évaluation & Benchmarking (Notebooks Jupyter)

L'évaluation est effectuée dans `main.ipynb` et `main_compare.ipynb` pour comparer les modèles et les prompts :

* **Scores NLP** : CIDEr, BLEU, METEOR, SPICE, CHAIR.
* **Fidélité visuelle** : Score **CHAIR** pour détecter les objets inventés.
* **Performance** : Temps d'inférence (Latence).

---

## 5. Installation et Configuration

### 5.1 Environnement (Conda)

```bash
conda create -n nlp_proj python=3.13
conda activate nlp_proj
pip install -r requirements.txt

```

### 5.2 Ollama et Modèles

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava
ollama pull moondream
ollama pull qwen2.5-vl:3b

```

### 5.3 Configuration Java (Metrics)

Requis pour le score METEOR via SDKMAN :

```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 8.0.402-amzn

```

---

## 6. Structure du Projet

```text
.
├── app.py                      # Point d'entrée Streamlit (Interface VQA)
├── main.ipynb                  # Expérimentations prompts + graphiques metrics
├── main_compare.ipynb          # Benchmark multi-modèles + graphiques metrics
├── .env                        # Configuration (DATASET_PATH, MODELS, JAVA_PATH)
├── requirements.txt            # Dépendances Python
├── core/
│   ├── Model.py                # Classe Model : Orchestre Ollama + mapping IDs COCO
│   └── rag.py                  # Pipeline CLIP complète (Init, DL, Seuil, Inférence)
├── evaluation/
│   ├── Scorer.py               # Orchestrateur (lance les calculs de scores)
│   ├── ChairScorer.py          # Logique spécifique aux hallucinations
│   └── MeteorScorer.py         # Liaison Python -> Java 8 pour METEOR
├── data/
│   ├── our_data/               # Captions_map (via dict_captions) & synonymes
│   └── test/                   # Échantillon d'images (dog.jpg, food.jpg, etc.)
├── utils/
│   ├── dict_captions.py        # Génération du dictionnaire de légendes
│   └── dl.py                   # Helpers pour téléchargements additionnels
└── temp_rag_images/            # Cache/Stockage temporaire pour le flux RAG

```

### Interface VQA streamlit

Lancez app.py puis streamlit run pour accéder à l'interface. Trouvez une démonstration en vidéo à ce lien : https://youtu.be/G_b3ViNHbyA

```
python .\src\app.py
streamlit run .\src\app.py
```

### Model.py

Ce fichier gère l'inférence avec Ollama. Il récupère les images et adapte le traitement des IDs : si le nom du fichier est un nombre, il le convertit en format COCO (entier), sinon il garde le nom d'origine (chaîne de caractères). La fonction `execute` renvoie deux dictionnaires : les descriptions générées par le modèle et les légendes réelles (Ground Truth) extraites du dataset.

### rag.py

Ce module contient toute la logique de recherche vectorielle. Il gère l'initialisation de CLIP, le téléchargement automatique du modèle et le calcul des probabilités. Pour la sélection, il identifie l'image avec la probabilité maximale, mais intègre un seuil de filtrage : si ce score est trop faible, aucune image n'est renvoyée pour éviter les erreurs de contexte. Le fichier est autonome et peut être utilisé sans l'interface Streamlit.

### Dossier Evaluation

Il regroupe les scripts de calcul des métriques (BLEU, METEOR, CIDEr, SPICE) et le score CHAIR pour quantifier les hallucinations. Le fichier `MeteorScorer.py` assure la liaison avec Java 8 (via le chemin défini dans le `.env`) pour exécuter les calculs de similarité sémantique.

### Notebooks de comparaison

* **main.ipynb** : Analyse l'impact des différents prompts sur les résultats. Il génère des graphiques pour visualiser l'évolution des scores selon la formulation des consignes.
* **main_compare.ipynb** : Compare les trois modèles (Llava, Moondream, Qwen2.5-VL) sur un même set d'images pour mesurer les différences de précision et de temps d'exécution.

### .env

Il centralise les paramètres locaux : le chemin absolu vers le dataset COCO, la liste des modèles Ollama disponibles, le modèle par défaut et le chemin vers l'exécutable Java 8 nécessaire aux métriques.

## 7. Analyse 

Voir les notebooks pour les analyses de resultats. src/main.ipnb et src/main_compare.ipynb
