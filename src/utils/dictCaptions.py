import json
import pickle
import os

JSON_PATH = "data/coco/annotations/captions_val2017.json"
OUTPUT_PATH = "src/utils/our_data/captions_map.pkl"


def captions_map_save():
    if os.path.exists(OUTPUT_PATH):
        print(f"{OUTPUT_PATH} already created.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    img_to_captions = {}

    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        
        # Initialize Key=image_id, Value=List, if picture encoutered for the first time
        if image_id not in img_to_captions:
            img_to_captions[image_id] = []
            
        img_to_captions[image_id].append(caption)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(img_to_captions, f)

if __name__=="__main__":
    captions_map_save()

