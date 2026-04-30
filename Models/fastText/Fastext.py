# code from: https://www.youtube.com/watch?v=Br-Ozg9D4mc

import fasttext

model_en = fasttext.load_model('C:\\Code')

model_en.get_nearest_neighbors("nationhood")
