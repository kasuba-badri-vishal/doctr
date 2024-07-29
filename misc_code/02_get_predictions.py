import os

os.chdir('./../')

languages = ['bengali', 'gujarati', 'punjabi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['printed', 'handwritten']

models = ['crnn_vgg16_bn']
languages = ['odia']
modalities = ['handwritten']
types_list = ['all']

data_dir = '/raid/ganesh/badri/RECOGNITION/data/'
device = '1'
batch_size = 512


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                try:
                    dataset_dir = os.path.join(data_dir,modality,lang)

                    command = f'python references/recognition/evaluate_pytorch_indic.py {model} --vocab {types}_{lang} --dataset {dataset_dir}/test/ --name {types}_{modality}_{model}_{lang} --device {device} -b {batch_size} --resume ./../models/{modality}/{model}/{types}_{modality}_{model}_{lang}.pt'
                except:
                    print(f'Error in {lang} {model}')
    # print(command)
    # exit()
    os.system(command)