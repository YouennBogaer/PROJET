import json
import string
import os

class ChairScorer:
    def __init__(self, instances_path, synonyms_path):
        """
        Initialise le calculateur CHAIR.
        :param instances_path: Chemin vers instances_val2017.json
        :param synonyms_path: Chemin vers coco_synonyms_SOTA.txt
        """
        self.coco_objects = self._load_coco_data(instances_path)
        self.synonyms = self._load_synonyms(synonyms_path)

    def _load_synonyms(self, path):
        """Charge le fichier texte de mapping mot -> catégorie"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier de synonymes introuvable : {path}")
            
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Le dernier mot est la catégorie cible (ex: 'traffic light')
                    # Les mots avant sont le synonyme (ex: 'stop light')
                    # Mais pour simplifier le format txt ci-dessus, on suppose souvent 
                    # mot_source -> mot_cible_coco (qui peut avoir des espaces)
                    
                    # Pour notre format simple : le dernier token est la target, le reste est la source
                    # Mais attention, 'traffic light' (COCO) a un espace.
                    # Astuce : On va charger intelligemment.
                    
                    # Cas simple du fichier fourni :
                    # col 1 = source, col 2...N = target
                    source = parts[0]
                    target = " ".join(parts[1:])
                    mapping[source] = target
        return mapping

    def _load_coco_data(self, instances_path):
        print(f"Chargement des annotations GT depuis {instances_path}...")
        with open(instances_path, 'r') as f:
            data = json.load(f)
        
        cats = {c['id']: c['name'] for c in data['categories']}
        img_to_objs = {}
        
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_name = cats[ann['category_id']]
            if img_id not in img_to_objs:
                img_to_objs[img_id] = set()
            img_to_objs[img_id].add(cat_name)
            
        return img_to_objs

    def _extract_objects(self, text):
        """
        Extrait les objets en gérant les mots simples et les mots composés (doublets).
        """
        # Nettoyage
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        
        found_objects = []
        skip_next = False
        
        # On parcourt les mots pour trouver des correspondances
        for i in range(len(words)):
            if skip_next:
                skip_next = False
                continue
                
            # 1. Essayer les mots composés (ex: "hot dog", "traffic light")
            if i < len(words) - 1:
                doublet = f"{words[i]} {words[i+1]}"
                if doublet in self.synonyms:
                    found_objects.append(self.synonyms[doublet])
                    skip_next = True # On saute le prochain mot car déjà utilisé
                    continue
            
            # 2. Essayer le mot simple
            w = words[i]
            if w in self.synonyms:
                found_objects.append(self.synonyms[w])
            # Gestion basique du pluriel 's' si pas dans le dict
            elif w.endswith('s') and w[:-1] in self.synonyms:
                found_objects.append(self.synonyms[w[:-1]])
                
        return found_objects

    def compute_score(self, model_responses):
        """
        Calcul les scores CHAIR.
        model_responses: Dictionnaire {img_id: ["caption"]}
        """
        hallucinated_objects = 0
        total_objects = 0
        
        for img_id, captions in model_responses.items():
            if not captions: continue
            
            caption = captions[0]
            
            gt_objects = self.coco_objects.get(img_id, set())
            generated_objects = self._extract_objects(caption)
            
            for obj in generated_objects:
                total_objects += 1
                if obj not in gt_objects:
                    hallucinated_objects += 1
            
        chair_score = (hallucinated_objects / total_objects) if total_objects > 0 else 0.0
        
        return chair_score
    
    def test(self, model_responses):
        hallucinated_objects = 0
        total_objects = 0
        
        for img_id, captions in model_responses.items():
            if not captions: continue
            
            # On prend la première caption par défaut
            caption = captions[0]
            
            gt_objects = self.coco_objects.get(img_id, set())
            print(f"GROUND TRUTH objects in image : {gt_objects}")
            generated_objects = self._extract_objects(caption)
            print(f"RESPONSE objects in image : {generated_objects}")
            
            has_hallucination = False
            for obj in generated_objects:
                total_objects += 1
                if obj not in gt_objects:
                    hallucinated_objects += 1
                    has_hallucination = True
            print(f"Image {img_id} has hallucination : {has_hallucination}; hallucinated object = {hallucinated_objects}/{total_objects}")
                
        chair_score = (hallucinated_objects / total_objects) if total_objects > 0 else 0.0
        
        return chair_score