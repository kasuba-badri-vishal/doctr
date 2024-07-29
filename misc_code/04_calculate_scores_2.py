import os
import pandas as pd
import jiwer
import json
# import fastwer


data_dir = './../../results/'

languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['printed', 'handwritten']
types_list = ['all', 'ihtr', 'akshara']


# languages = ['malayalam']
modalities = ['printed']
models = ['crnn_vgg16_bn']
types_list = ['all']


results_dict = {}


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                # try:
                df = pd.read_csv(data_dir + f'{types}_{modality}_{model}_{lang}.csv', names=['gt','pred'], sep=' ')
                # print(df.head())
                
                # convert to string
                df['pred'] = df['pred'].astype(str)
                df['gt'] = df['gt'].astype(str)
                final_predictions = df['pred'].tolist()
                final_ground_truths = df['gt'].tolist()

                # CRR = 100 - fastwer.score(final_predictions, final_ground_truths, char_level=True)
                # WRR = 100 - fastwer.score(final_predictions, final_ground_truths)
                
                words_output = jiwer.process_words(final_ground_truths, final_predictions)
                chars_output = jiwer.process_characters(final_ground_truths, final_predictions)
                WRR = 100 - words_output.wer
                CRR = 100 - chars_output.cer
                
                print(f'{lang} {model} CRR: {CRR} WRR: {WRR}')
                
                results_dict[f'{lang}_{model}_{modality}_{types}'] = [CRR, WRR]
            # except:
            #     print(f'Error in {lang} {model}')
                    
           
# Save results in csv file
results_df = pd.DataFrame(results_dict).T

results_df.columns = ['CRR', 'WRR']
results_df.to_csv('results.csv')
           
                    
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)