import os
import pandas as pd
import fastwer
import json


data_dir = './../../results/'

languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['printed', 'handwritten']
types_list = ['all', 'ihtr', 'akshara']


# languages = ['bengali']
modalities = ['printed']
models = ['crnn_vgg16_bn']
types_list = ['all']


results_dict = {}


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                try:
                    df = pd.read_csv(data_dir + f'{types}_{modality}_{model}_{lang}.csv', names=['gt','pred'], sep=' ')
                    # print(df.head())
                    
                    # convert to str
                    df['pred'] = df['pred'].astype(str)
                    df['gt'] = df['gt'].astype(str)
                    
                    final_predictions = df['pred'].tolist()
                    final_ground_truths = df['gt'].tolist()
            
                    CRR = 100 - fastwer.score(final_predictions, final_ground_truths, char_level=True)
                    WRR = 100 - fastwer.score(final_predictions, final_ground_truths)
                    
                    CRR = round(CRR,2)
                    WRR = round(WRR,2)
                    
                    print(f'{lang} {model} CRR: {CRR} WRR: {WRR}')
                    
                    results_dict[f'{lang}_{model}_{modality}_{types}'] = [CRR, WRR]
                except:
                    print(f'Error in {lang} {model}')
                    
                    
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)