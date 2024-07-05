import os
import pandas as pd
import json
import imghdr
from PIL import Image

def get_word_lengths(merged_df_set, vocab):
    max_word_length = 0
    min_word_length = 100000
    avg_word_length = 0
    for word in merged_df_set:
        word = str(word)
        if len(word) >= max_word_length:
            max_word_length = len(word)
        elif len(word) < min_word_length:
            min_word_length = len(word)
        
        avg_word_length += len(word)
    avg_word_length = avg_word_length / len(merged_df_set)
    
    vocab['max_word_length'] = max_word_length
    vocab['min_word_length'] = min_word_length
    vocab['avg_word_length'] = round(avg_word_length, 2)    
    return vocab

def get_word_dimensions(merged_df, vocab):
    # calculate avg dimensions
    merged_df['width'] = merged_df['dimensions'].apply(lambda x: x[0])
    merged_df['height'] = merged_df['dimensions'].apply(lambda x: x[1])
    merged_df['size'] = merged_df['dimensions'].apply(lambda x: x[2])
    
    vocab['avg_width'] = round(merged_df['width'].mean(), 2)
    vocab['avg_height'] = round(merged_df['height'].mean(), 2)
    vocab['avg_size'] = round(merged_df['size'].mean(), 2)
    
    vocab['max_width'] = int(merged_df['width'].max())
    vocab['max_height'] = int(merged_df['height'].max())
    vocab['max_size'] = int(merged_df['size'].max())
    
    vocab['min_width'] = int(merged_df['width'].min())
    vocab['min_height'] = int(merged_df['height'].min())
    vocab['min_size'] = int(merged_df['size'].min())
    
    #max area
    merged_df['area'] = merged_df['width'] * merged_df['height']
    vocab['max_area'] = int(merged_df['area'].max())
    vocab['min_area'] = int(merged_df['area'].min())
    
    merged_df['ratio'] = merged_df['width'] / merged_df['height']
    vocab['max_ratio'] = round(merged_df['ratio'].max(), 2)
    vocab['min_ratio'] = round(merged_df['ratio'].min(), 2)
    vocab['avg_ratio1'] = round(merged_df['ratio'].mean(), 2)
    vocab['avg_ratio2'] = round(merged_df['width'].mean() / merged_df['height'].mean(), 2)
    
    return vocab

def get_image_dimensions(image_path):
    try:
        image_type = imghdr.what(image_path)
        if image_type:
            with open(image_path, 'rb') as img_file:
                img_file.seek(0, os.SEEK_END)
                file_size = img_file.tell()
                img_file.seek(0)
                with Image.open(img_file) as img:
                    width, height = img.size
                    return width, height, file_size
        else:
            
            print("Not a valid image file:", image_path)
            # return None, None, None
    except IOError:
        print("Unable to open image file:", image_path)
        # return None, None, None



langs = [ 'bengali', 'gujarati', 'hindi', 'kannada', 'malayalam', 'odia', 'punjabi', 'tamil', 'telugu', 'urdu']
langs = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'marathi', 'manipuri', 'odia', 'tamil', 'telugu', 'urdu']
modality = 'printed'
data_dir = '/raid/ganesh/badri/RECOGNITION/data/'

for lang in langs:
    
    if not os.path.exists(os.path.join(data_dir, modality, lang, 'stats')):
        os.makedirs(os.path.join(data_dir, modality, lang, 'stats'))
        
    merged_df = pd.DataFrame()
    
    vocab = {}
    
    for sets in ['test','val','train']:
        input_file  = os.path.join(data_dir, modality, lang, sets, sets + '_gt.txt')
        print(input_file)
        df = pd.read_csv(input_file, delimiter='\t', names=['file','label'], engine="python", quoting=3, encoding='utf-8')
        print(df.head())
        
        # count value counts of label with frequency and save
        df['label'].value_counts().to_csv(os.path.join(data_dir, modality, lang, 'stats', sets + '_label_freq.csv'))
        
        vocab[sets] = set()
        df['label'].apply(lambda x: [vocab[sets].add(i) for i in str(x)])
        
        
        input_data_dir = os.path.join(data_dir, modality, lang, sets + '/')
        df['dimensions'] = df['file'].apply(lambda x : get_image_dimensions(input_data_dir + x))
 
        merged_df = pd.concat([merged_df, df])
        
    merged_df_set = set(merged_df['label'])
    df_set = set(df['label'])

    
    words = merged_df_set - df_set
    with open(os.path.join(data_dir, modality, lang, 'stats', 'oov.txt'), 'w') as f:
        for item in words:
            f.write("%s\n" % item)
                 
       
    vocab['all'] = set()
    merged_df['label'].apply(lambda x: [vocab['all'].add(i) for i in str(x)])
    
    vocab['oov'] = vocab['all'] - vocab['train']
    vocab['oov_test'] = vocab['test'] - vocab['train']
    vocab['oov_val'] = vocab['val'] - vocab['train']
    
    for key, value in vocab.items():
        current = sorted(list(value))
        vocab[key] = ''.join(current)
        
        
    vocab = get_word_lengths(merged_df_set, vocab)
    vocab = get_word_dimensions(merged_df, vocab)
    
    with open(os.path.join(data_dir, modality, lang, 'stats', 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
    
    # store all unique values with frequency
    merged_df['label'].value_counts().to_csv(os.path.join(data_dir, modality, lang, 'stats', 'all_label_freq.csv'))

        