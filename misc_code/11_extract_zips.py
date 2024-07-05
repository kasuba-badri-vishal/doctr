import os
import zipfile
from tqdm import tqdm

def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
        
langs = [ 'bengali', 'gujarati', 'hindi', 'kannada', 'malayalam', 'odia', 'punjabi', 'tamil', 'telugu', 'urdu']
langs = ['assamese', 'bengali', 'gujarati',  'hindi', 'kannada', 'malayalam', 'marathi', 'manipuri', 'odia', 'punjabi', 'tamil', 'telugu', 'urdu']

modality = 'printed'

data_dir = '/raid/ganesh/badri/RECOGNITION/data/'

for sets in ['test','val','train']:
    for lang in tqdm(langs):
        try:
            input_dir  = os.path.join(data_dir, modality, lang, sets + '.zip')
            output_dir = os.path.join(data_dir, modality, lang)
            extract_zip(input_dir, output_dir)
            os.remove(input_dir)
        except:
            print(sets, lang)

