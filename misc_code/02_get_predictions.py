import os
os.chdir('./../')

from config import *


# Override the config variables
languages = ['tamil']
models = ['crnn_vgg16_bn', 'master']
modalities = ['handwritten']
types_list = ['ihtr']
dataset = 0


# Model Training Parameters
device = "1"
batch_size = 512


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                dataset_dir = os.path.join(data_dir,modality,datasets_collection[modality][dataset], lang)
                model_name = f"{types}_{modality}_{model}_{lang}"
                model_dir = os.path.join(models_dir,modality, model)
                result_dir = os.path.join(results_dir,modality, model)
                
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                # exit()
                command = f"python references/recognition/evaluate_pytorch_indic.py {model} --dataset {dataset_dir}/test/ --name {model_name} --device {device} --vocab {types}_{lang} -b {batch_size} --results_dir {result_dir} --resume {model_dir}/{model_name}.pt"
                os.system(command)