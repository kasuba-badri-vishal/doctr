import os
os.chdir('./../')

from config import *


# Override the config variables
languages = ['tamil']
models = ['crnn_vgg16_bn']
modalities = ['handwritten']
types_list = ['ihtr']
dataset = 0




# Model Training Parameters
device = "1"
batch_size = 512
epochs = 30
learning_rate = 0.0001


resume = True
wandb_log = True


for lang in languages:
    for types in types_list:
        for modality in modalities:
            for model in models:
                dataset_dir = os.path.join(data_dir,modality,datasets_collection[modality][dataset], lang)
                model_dir = os.path.join(models_dir,modality, model)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                model_name = f"{types}_{modality}_{model}_{lang}"
                try:
                    if(resume):
                        command = f"python references/recognition/train_pytorch_indic.py {model} --train_path {dataset_dir}/train/ --val_path {dataset_dir}/val/ --name {model_name} --device {device} --vocab {types}_{lang} -b {batch_size} --epochs {epochs} --lr {learning_rate} --model_dir {model_dir} --resume {model_dir}/{model_name}.pt"
                    else:
                        command = f"python references/recognition/train_pytorch_indic.py {model} --train_path {dataset_dir}/train/ --val_path {dataset_dir}/val/ --name {model_name} --device {device} --vocab {types}_{lang} -b {batch_size} --epochs {epochs} --lr {learning_rate} --model_dir {model_dir}"
                    
                    if os.system(command) != 0:
                        break
                except:
                    print(f"Error training {model_name}, {lang}, {modality}, {model}")
                    exit()
