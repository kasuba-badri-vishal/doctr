import os

langs = ['assamese', 'bengali', 'gujarati', 'punjabi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'oriya', 'tamil', 'telugu', 'urdu']

# langs = ['hindi']
lang2 = [word.capitalize() for word in langs]

print(lang2)

curr_dir = os.getcwd()

data_dir = '/raid/ganesh/badri/RECOGNITION/data/iiit_synthetic/'

for lang, cap in zip(langs, lang2):
    # print(lang, cap)
    
    if not os.path.exists(f'{data_dir}{lang}/'):
        os.makedirs(f'{data_dir}{lang}/')
    
    os.chdir(f'{data_dir}{lang}/')
    for sets in ['train']:
        
        os.system(f'curl --insecure -o {sets}.zip https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in//public/printed/phase-0/v0.5/{lang}/IIIT-Synthetic-R-{cap}/{sets}.zip')
    os.chdir(curr_dir)
    # exit()
    
# import os

# langs = ['assamese', 'bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']



# for lang in langs:
#     os.chdir(f'./{lang}/')
#     for sets in ['train', 'val', 'test']:
        
#         os.system(f'unzip {sets}.zip')
#         os.system(f'rm {sets}.zip')
#     os.chdir('../')
#     # exit()