from pathlib import Path
import ollama

class Model():
   def __init__(self, prompts, imgs_path, coco_captions):
       self.imgs_path = imgs_path
       self.prompts = prompts
       self.coco_captions = coco_captions

   def execute(self, prompt_id, freq_print=10):
      """
      freq_print (default 10) : if 0 then does not print.
      """

      # Récupère les ids des images dans l'ordre de images_to_process
      images_ids = [int(Path(path).stem) for path in self.imgs_path]

      print(f"--- Starting analysis. Selected images : {len(images_ids)} ---")

      # Dict of model responses per image (key = image id, value = list of responses)
      model_responses = {}

      num_img = len(self.imgs_path)

      for n,img_path in enumerate(self.imgs_path):
         img_id = images_ids[n]

         try:
            reponse = ollama.chat(
                  model='llava',
                  messages=[{
                     'role': 'user',
                     'content': self.prompts[prompt_id],
                     'images': [img_path]
                  }]
            )
            
            description = reponse['message']['content']
            description = description.strip().strip('"')
            if (freq_print > 0 and n%freq_print == 0):
               print(f"Analyse {n+1}/{num_img} : {img_id}, {img_path}")
               print(f"    {description}")

            if img_id not in model_responses:
                  model_responses[img_id] = []
            model_responses[img_id].append(description) 

         except Exception as e:
            print(f"Erreur sur l'image {img_path}: {e}")   

      # We create a dictionnary of ground truth captions for the specific images tested below
      gt_captions_dict = {img_id: self.coco_captions.get(img_id, "Pas de légende trouvée") for img_id in model_responses}

      return model_responses, gt_captions_dict
