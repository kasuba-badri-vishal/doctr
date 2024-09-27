import os
import json
import pandas as pd

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





languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
languages = ['bengali', 'gujarati', 'punjabi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['printed', 'handwritten']
types_list = ['all', 'ihtr', 'akshara']


# languages = ['malayalam']
modalities = ['handwritten']
models = ['crnn_vgg16_bn']
types_list = ['all']


data_dir = './../../results/'
scores_dir = './../../scores/'


results_dict = {}


for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                
                dataset_dir = os.path.join(data_dir,modality,model)
                score_data_dir = os.path.join(scores_dir,modality,model)
            
                df = pd.read_csv(dataset_dir + f'/{types}_{modality}_{model}_{lang}.csv', names=['gt','pred'], sep=' ')
                # print(df.head())
                
                # convert to string
                df['pred'] = df['pred'].astype(str)
                df['gt'] = df['gt'].astype(str)
                
                df['edits'] = df.apply(lambda row: edit_distance(row['pred'], row['gt']), axis=1)
                df['length'] = df['gt'].apply(lambda row: len(row))
                df['wer'] = df.apply(lambda row: 1 if (row['pred']!=row['gt']) else 0, axis=1)

                cer = df['edits'].sum()/df['length'].sum()
                wer = df['wer'].mean()

                WRR = (1 - wer) * 100
                CRR = (1 - cer) * 100
                
                
                print(f'{lang} {model} CRR: {CRR} WRR: {WRR}')
                
                results_dict[f'{lang}_{model}_{modality}_{types}'] = [CRR, WRR]

                if not os.path.exists(score_data_dir):
                    os.makedirs(score_data_dir)

                df.to_csv(score_data_dir + f'/{types}_{modality}_{model}_{lang}_metrics.csv', index=False, sep=' ')
                    

# Save results in csv file
results_df = pd.DataFrame(results_dict).T

results_df.columns = ['CRR', 'WRR']
results_df.to_csv('results.csv')
           
                    
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)
