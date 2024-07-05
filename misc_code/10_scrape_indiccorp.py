import os
import subprocess

# Get languages from lists
link = 'https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/'


languages = ['as', 'ml', 'bd', 'mni', 'bn', 'mr', 'dg', 'ne', 'en', 'or', 'gom', 'pa', 'gu', 'sa', 'hi', 'sat', 'kha', 'sd', 'kn', 'ta', 'ks', 'te', 'mai']

#Full forms of languages
languages_full = ['assamese', 'malayalam', 'bodo', 'manipuri', 'bengali', 'marathi', 'dogri', 'nepali', 'english', 'oriya', 'konkani', 'punjabi', 'gujarati', 'sanskrit', 'hindi', 'santali', 'khasi', 'sindhi', 'kannada', 'tamil', 'kashmiri', 'telugu', 'maithili']

low_resource = ['bodo', 'manipuri', 'dogri', 'goan konkani', 'santali', 'khasi', 'sindhi', 'kashmiri', 'maithili']
low_resource = ['konkani']


langs = {
    'as' : 'assamese',
    'ml' : 'malayalam',
    'bd' : 'bodo',
    'mni' : 'manipuri',
    'bn' : 'bengali',
    'mr' : 'marathi',
    'dg' : 'dogri',
    'ne' : 'nepali',
    'en' : 'english',
    'or' : 'oriya',
    'gom' : 'konkani',
    'pa' : 'punjabi',
    'gu' : 'gujarati',
    'sa' : 'sanskrit',
    'hi' : 'hindi',
    'sat' : 'santali',
    'kha' : 'khasi',
    'sd' : 'sindhi',
    'kn' : 'kannada',
    'ta' : 'tamil',
    'ks' : 'kashmiri',
    'te' : 'telugu',
    'mai' : 'maithili'
}


unicode_ranges = {
    'assamese' : (0x0980, 0x09FF),
    'malayalam' : (0x0D00, 0x0D7F),
    'bodo' : (0x0C80, 0x0CFF),
    'manipuri' : (0x0900, 0x097F),
    'bengali' : (0x0980, 0x09FF),
    'marathi' : (0x0900, 0x097F),
    'dogri' : (0x0A80, 0x0AFF),
    'nepali' : (0x0900, 0x097F),
    'english' : (0x0000, 0x007F),
    'oriya' : (0x0B00, 0x0B7F),
    'goan konkani' : (0x0900, 0x097F),
    'punjabi' : (0x0A00, 0x0A7F),
    'gujarati' : (0x0A80, 0x0AFF),
    'sanskrit' : (0x0900, 0x097F),
    'hindi' : (0x0900, 0x097F),
    'santali' : (0x0C80, 0x0CFF),
    'khasi' : (0x0A00, 0x0A7F),
    'sindhi' : (0x0A80, 0x0AFF),
    'kannada' : (0x0C80, 0x0CFF),
    'tamil' : (0x0B80, 0x0BFF),
    'kashmiri' : (0x0900, 0x097F),
    'telugu' : (0x0C00, 0x0C7F),
    'maithili' : (0x0900, 0x097F)
}


data_dir = '/raid/ganesh/badri/RECOGNITION/data/IndicCorpV2/'

for lang in languages:
    
    if(langs[lang] in low_resource):
        curl_command = 'curl -o ' + data_dir + langs[lang] + '.txt ' + link + lang + '.txt'
        # print(curl_command)
        # exit()
        os.system(curl_command)
        print("Done with " + lang + ".txt\n")