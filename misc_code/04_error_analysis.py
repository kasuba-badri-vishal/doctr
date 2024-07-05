import pandas as pd

data_dir = './../results/'

languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['printed', 'handwritten']
types_list = ['all', 'ihtr', 'akshara']


languages = ['hindi']
modalities = ['printed']
models = ['crnn_vgg16_bn']
types_list = ['all']


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                df = pd.read_csv(data_dir + f'{types}_{modality}_{model}_{lang}.csv', names=['gt','pred'], sep=' ')
                print(df.shape)
                temp = df[df['gt']!=df['pred']]
                print(temp)
                