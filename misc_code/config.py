# All 22 Indian languages
languages = ['assamese', 'bengali', 'bodo', 'dogri', 'gujarati', 'hindi','kannada','kashmiri', 'konkani', 'maithili', 'malayalam', 'manipuri', 'marathi', 'nepali', 'odia', 'punjabi', 'sanskrit', 'santali', 'sindhi', 'tamil', 'telugu', 'urdu']

# All 13 Indic Languages
languages = ['assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu']

# All Models 
models = ['crnn_vgg16_bn', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'master', 'parseq', 'sar_resnet31', 'vitsr_base', 'vitstr_small']

# All Modality
modalities = ['handwritten', 'printed', 'scene_text', 'iiit_synthetic']

# All Types
vocabs = ['all', 'ihtr', 'akshara', 'unicode', 'scene_text']

# Datasets
datasets_collection = {
    'handwritten': {0 : "iiit_indic_hw_words", 1 : "iiit_indic_uc", 2 : "iiit_hw_english_word", 3 : "iam"},
    'printed': {0 : "akshara", 1 : "iiit_synthetic"},
    'scene_text': {0 : "indicstr12", 1 : "iiit_synthetic"}
}


data_dir = '/data/BADRI/DATASETS/BENCHMARK/RECOGNITION/'
models_dir = '/data/BADRI/RESEARCH/RECOGNITION/models/'
results_dir = '/data/BADRI/RESEARCH/RECOGNITION/results/'
scores_dir = '/data/BADRI/RESEARCH/RECOGNITION/scores/'
