# code from: https://pypi.org/project/fasttext/#installation

import fasttext

# 1) split the data before training
# 2) train a first fastText model
# 3) use gridSearch to build another model and then compare
# 4) begin analysis of the vector space with get_nearest_neighbors()


## 1) SPLIT of the DATA

## 2) TRAINING

# Skipgram model :
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# or, cbow model :
model = fasttext.train_unsupervised('data.txt', model='cbow')


# Save trained model object by calling the function save_model
model.save_model("model_filename.bin")

# Retrieve saved model using function load_model
model = fasttext.load_model("model_filename.bin")



## 3) GRIDSEARCH
from sklearn.model_selection import GridSearchCV

# Define hyperparameters potential values before GridSearch
parameters = {
    'model': ['cbow','skipgram']
    'lr': [0.05],
    'dim': [100, 200],
    'ws': [5],
    'epoch': [5],
    'minCount': [5],
    'minn': [3],
    'maxn': [6],
    'neg': [5],
    'wordNgrams': [1],
    'loss': [ns, hs, softmax, ova],
    'bucket': [20000000],
    'lrUpdateRate': [100],
    't': [0.0001],
    'verbose': [2],
}

svc = svm.SVC()
clf_grid_search = GridSearchCV(svc, parameters)

# Create an instance of fastText with GridSearch and fit the data
clf_grid_search.fit(X_train, y_train)


## 4) VECTOR SIMILARITY SEARCH

model.get_nearest_neighbors("nationhood")
