import json
import os
import pandas as pd
import fastwer
os.chdir('./../')

modality_codes = {
    "printed": "P",
    "handwritten": "HW",
    "scenetext": "ST"
}

lang_codes = {
    "assamese": "ASA",
    "bengali": "BN",
    "gujarati": "GU",
    "hindi": "HI",
    "kannada": "KN",
    "malayalam": "ML",
    "marathi": "MR",
    "odia": "ORI",
    "punjabi": "PA",
    "tamil": "TA",
    "telugu": "TE",
    "urdu": "UR"
}

data_codes = ['101', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166']

# data_codes = ['101','151','164']

languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']
models = ['crnn_vgg16_bn', 'master', 'parseq', 'vitsr_base', 'vitstr_small', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31']
modalities = ['handwritten', 'printed', 'iiit_synthetic']
types_list = ['all', 'ihtr', 'akshara']

device = '5'


# Current usage
# languages = ['assamese', 'bengali', 'gujarati', 'punjabi', ]
models = ['crnn_vgg16_bn']
modalities = ['printed']
types_list = ['all']


data_dir = "/raid/ganesh/badri/RECOGNITION/data/bhashini/"
results_dir = "/raid/ganesh/badri/RECOGNITION/pipeline/results/"

results_dict = {}

for lang in languages:
    for model in models:
        for modality in modalities:
            for types in types_list:
                for data_code in data_codes:
                    try:
                        files_dir = "BM" + "_" + modality_codes[modality] + "_" + lang_codes[lang] + "_" + str(data_code)
                        dataset_dir = os.path.join(data_dir,modality,"extracts",files_dir)
                        print(dataset_dir)
                        if(os.path.exists(dataset_dir)):
                            command = f'python references/recognition/evaluate_pytorch_indic.py {model} --vocab {types}_{lang} --dataset {dataset_dir}/ --name {types}_{modality}_{model}_{lang}_{files_dir} --device {device} -b 1024 --resume ./../models/{types}_{modality}_{model}_{lang}.pt'
                            os.system(command)
                            
                            df = pd.read_csv(results_dir + f'{types}_{modality}_{model}_{lang}_{files_dir}.csv', names=['gt','pred'], sep=' ')
                            # print(df.head())
                            
                            results = {}
                            results['size'] = df.shape[0]
                            
                            # replace nan with empty strings
                            df['gt'] = df['gt'].fillna('')
                            df['pred'] = df['pred'].fillna('')
                            
                            final_predictions = df['pred'].tolist()
                            final_ground_truths = df['gt'].tolist()

                            CRR = 100 - fastwer.score(final_predictions, final_ground_truths, char_level=True)
                            WRR = 100 - fastwer.score(final_predictions, final_ground_truths)
                            
                            # save to 2 decimal places
                            results['CRR'] = round(CRR, 2)
                            results['WRR'] = round(WRR, 2)
                            
                            print(f'{lang} {model} CRR: {CRR} WRR: {WRR}')
                            
                            results_dict[f'{lang}_{model}_{modality}_{types}_{files_dir}'] = results
                    except Exception as e:
                        print(f'Error in {lang} {model}')
                        # print error
                        print(e)
                        
                        
print(results_dict)

#save json
import json

with open('bhashini_results.json', 'w') as fp:
    json.dump(results_dict, fp, indent=4, ensure_ascii=False)
    
    
    