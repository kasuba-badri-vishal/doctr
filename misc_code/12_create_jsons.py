# import os
import pandas as pd
import json

# langs = ['bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
langs = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'marathi', 'manipuri', 'odia', 'tamil', 'telugu', 'urdu']
data_dir = '/raid/ganesh/badri/RECOGNITION/data/printed'


for lang in langs:
    for sets in ['val','test','train']:
        
        data = {}
        
        with open(f'{data_dir}/{lang}/{sets}/{sets}_gt.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    filename, word= line.split('\t')
                    filename = filename[7:]
                except:
                    print(line)
                    exit()
                data[filename] = word.strip()
        
        
        # # df = pd.read_csv(f'{data_dir}/{lang}/{sets}/{sets}_gt.txt',sep='\t', names=['filename','word'])
        # data = df.set_index('filename').to_dict()['word']
        
        #save json
        with open(f'{data_dir}/{lang}/{sets}/labels.json', 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        # exit()