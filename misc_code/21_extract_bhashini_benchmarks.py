# Extract images from the base64 encoded strings in the json file

# Json file reads: 
import os
import json
import base64

def extract_images(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    if not os.path.exists(output_dir + "/images"):
        os.makedirs(output_dir + "/images")
    
    count = 1
    final_data = {}
    for data_item in data:
        with open(f'{output_dir}/images/{str(count)}.png', 'wb') as f:
            f.write(base64.b64decode(data_item["image"]))
        final_data[str(count) + ".png"] = data_item["gt"]
        count +=1
    
    with open(f'{output_dir}/labels.json', 'w') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    # for key in data:
    #     with open(f'{output_dir}/{key}.png', 'wb') as f:
    #         f.write(base64.b64decode(data[key]))


modalities = ['scenetext']

data_dir = '/raid/ganesh/badri/RECOGNITION/data/bhashini/'

for modality in modalities:

    files_dir = f"{data_dir}{modality}/jsons_data/"
    data_dir = f"{data_dir}{modality}/extracts/"

    for file in os.listdir(files_dir):
        if file.endswith('.json'):
            extract_images(files_dir + file, data_dir + file.split('.')[0])

# extract_images('./../handwritten/Phase-0_BM_HW_TE_151.json', './../handwritten2/BM_HW_TE_151/')