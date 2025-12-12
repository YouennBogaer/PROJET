import string
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.ChairScorer import ChairScorer

class Scorer:
    def __init__(self, path_instances, path_synonyms):
        self.path_instances = path_instances
        self.path_synonyms = path_synonyms

    def sanitize_text(self, text):
      text = text.lower()
      text = text.replace('\n', ' ').replace('\r', ' ') 
      text = text.translate(str.maketrans('', '', string.punctuation))
      text = " ".join(text.split())
      return text

    def sanitize_dict(self, data_dict):
        """
        Applique le nettoyage sur tout un dictionnaire {id: [liste_captions]}
        """
        clean_dict = {}
        for img_id, captions in data_dict.items():
            clean_dict[img_id] = [self.sanitize_text(c) for c in captions]
        return clean_dict

    def compute_bleu(self, gts, res):
        """
        BLEU (Bilingual Evaluation Understudy)
        Compte le nombre de séquences de mots (n-grammes) qui apparaissent à la fois dans la légende générée et dans la référence. 
        Mesure de précision. Peu fiable car ne reconnaît pas les synonymes.

        PARAMETERS : 

         gts: Dictionnaire {image_id: [liste_de_captions_reference]}
         res: Dictionnaire {image_id: [liste_de_captions_generees]}
        """
        print(f"Calcul de Bleu...")

        gts = self.sanitize_dict(gts)
        res = self.sanitize_dict(res)

        eval_result = {}

        scorer, methods = (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        
        # global score, individual scores 
        score, _ = scorer.compute_score(gts,res)

        for m, s in zip(methods,score):
            eval_result[m] = s

        return eval_result
    
    # TODO Bibliothèque galère à utiliser sur windows à cause des passage en java (à tester avec JAVA 8)
    def compute_meteor(self,gts,res):
        """
        METEOR (Metric for Evaluation of Translation with Explicit ORdering)
        Similaire à BLEU. Aligne les mots en utilisant non seulement la correspondance exacte,
        mais aussi les synonymes (via WordNet) et les racines des mots (stemming).
        Permet de mesurer la richesse du vocabulaire.

        PARAMETERS : 

         gts: Dictionnaire {image_id: [liste_de_captions_reference]}
         res: Dictionnaire {image_id: [liste_de_captions_generees]}
        """
        return None
    
    def compute_cider(self,gts,res):
        """
        CIDEr (Consensus-based Image Description Evaluation)
        Utilise une approche TF-IDF pour donner plus de poids aux mots importants et moins aux mots courants. 
        Elle mesure à quel point la légende générée capture le "consensus" des références humaines.
        Métrique standard dans les compétitions de captioning (comme COCO)

        PARAMETERS : 

         gts: Dictionnaire {image_id: [liste_de_captions_reference]}
         res: Dictionnaire {image_id: [liste_de_captions_generees]}
        """
        print(f"Calcul de CIDEr...")

        gts = self.sanitize_dict(gts)
        res = self.sanitize_dict(res)

        scorer = Cider()

        score, _ = scorer.compute_score(gts,res)

        return score
    
    def compute_spice(self, gts, res):
        """
        SPICE (Semantic Propositional Image Caption Evaluation)
        transforme les captions (ground truth et response) en "graphes de scène" et compare ensuite ces graphes.
        Exemple : elle vérifie si le modèle a bien détecté (Chat) -> [est sur] -> (Tapis).
        """
        print(f"Calcul de SPICE...")

        first_id = next(iter(res))
        if not res[first_id] or len(res[first_id][0].strip()) == 0:
            print("ATTENTION: Caption vide détectée. SPICE annulé pour éviter le crash.")
            return 0.0, []
    
        try:
         spice_scorer = Spice()
        except Exception as e:
         print(f"Erreur à l'init de SPICE (Java manquant ?): {e}")
         return None

        gts = self.sanitize_dict(gts)
        res = self.sanitize_dict(res)

        avg_score, scores_detailed = spice_scorer.compute_score(gts, res)
        return avg_score, scores_detailed
        
    
    def compute_chair(self,res):
        """
        CHAIR (Caption Hallucination Assessment with Image Relevance)
        C'est une métrique spécialisée pour détecter les hallucinations. 
        Elle calcule le pourcentage d'objets mentionnés dans la légende qui ne sont pas présents dans l'image 
        (basé sur la liste d'objets connus par exemple grace une segmentation).
        """
        print(f"Calcul de CHAIR...")

        res = self.sanitize_dict(res)
        
        scorer = ChairScorer(self.path_instances,self.path_synonyms)
        score = scorer.compute_score(res)
        return score
    
    def compute_scores(self,gts,res):
        eval_result = {}

        # CIDEr
        eval_result["CIDEr"] = self.compute_cider(gts,res)

        # BLEU
        bleu_result = self.compute_bleu(gts,res)
        eval_result = eval_result | bleu_result

        # SPICE
        avg_spice,_ = self.compute_spice(gts,res)
        eval_result["SPICE"] = avg_spice

        # CHAIR
        eval_result["CHAIR"] = self.compute_chair(res)

        return eval_result

