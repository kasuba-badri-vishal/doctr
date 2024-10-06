import os
import json
import pandas as pd

from config import *

def edit_distance(predicted: str, ground_truth: str) -> float:
    len_pred = len(predicted)
    len_gt = len(ground_truth)
    dp = [[0] * (len_gt + 1) for _ in range(len_pred + 1)]

    for i in range(len_pred + 1):
        dp[i][0] = i
    for j in range(len_gt + 1):
        dp[0][j] = j
        
    for i in range(1, len_pred + 1):
        for j in range(1, len_gt + 1):
            if predicted[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                
    return dp[len_pred][len_gt]





languages = ['tamil']
modalities = ['handwritten']
models = ['crnn_vgg16_bn', 'master']
types_list = ['ihtr']




results_dict = {}


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                
                file_name = f'{types}_{modality}_{model}_{lang}.csv'
                dataset_dir = os.path.join(results_dir,modality,model, file_name)
                score_data_dir = os.path.join(scores_dir,modality,model)
            
                df = pd.read_csv(dataset_dir, names=['gt','pred'], sep='\t')
                # print(df.head())
                
                print(df.head())
                
                # convert to string
                df['pred'] = df['pred'].astype(str)
                df['gt'] = df['gt'].astype(str)
                
                df['edits'] = df.apply(lambda row: edit_distance(row['pred'], row['gt']), axis=1)
                df['length'] = df['gt'].apply(lambda row: len(row))
                df['wer'] = df.apply(lambda row: 1 if (row['pred']!=row['gt']) else 0, axis=1)

                cer = df['edits'].sum()/df['length'].sum()
                wer = df['wer'].mean()

                WRR = round(wer * 100, 2)
                CRR = round(cer * 100, 2)
                
                
                print(f'{lang} {model} CRR: {CRR} WRR: {WRR}')
                
                results_dict[f'{lang}_{model}_{modality}_{types}'] = [CRR, WRR]

                if not os.path.exists(score_data_dir):
                    os.makedirs(score_data_dir)

                df.to_csv(score_data_dir + file_name, index=False, sep='\t')
                    

# Save results in csv file
results_df = pd.DataFrame(results_dict).T

results_df.columns = ['CRR', 'WRR']
results_df.to_csv('results.csv')
           
                    
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)