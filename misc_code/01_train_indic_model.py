import os
os.chdir('./../')


languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['handwritten', 'printed', 'iiit_synthetic']
types_list = ['all', 'ihtr', 'akshara', 'unicode']



languages = ['odia', 'tamil', 'telugu', 'urdu']
models = ['master']
modalities = ['printed']
types_list = ['all']


data_dir = '/raid/ganesh/badri/RECOGNITION/data/'
models_dir = '/raid/ganesh/badri/RECOGNITION/pipeline/models'


device = '3'
batch_size = 512
epochs = 30
learning_rate = 0.001


resume = True


for modality in modalities:
    for model in models:
        model_dir = os.path.join(models_dir,modality, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for lang in languages:
            for types in types_list:
                try:
                    dataset_dir = os.path.join(data_dir,modality,lang)
                    if(resume) and (os.path.exists(f'{model_dir}/{types}_{modality}_{model}_{lang}.pt')):
                        command = f'python references/recognition/train_pytorch_indic.py {model} --train_path {dataset_dir}/train/ --val_path {dataset_dir}/val/ --name {types}_{modality}_{model}_{lang} --device {device} --vocab {types}_{lang} -b {batch_size} --epochs {epochs} --lr {learning_rate} --wb --resume {model_dir}/{types}_{modality}_{model}_{lang}.pt'
                    else:
                        command = f'python references/recognition/train_pytorch_indic.py {model} --train_path {dataset_dir}/train/ --val_path {dataset_dir}/val/ --name {types}_{modality}_{model}_{lang} --device {device} --vocab {types}_{lang} -b {batch_size} --epochs {epochs} --lr {learning_rate} --wb'
                    
                    # print(command)
                    # exit()
                    if os.system(command) != 0:
                        break
                    # os.system(command)
                except:
                    exit()
