import pandas as pd
import nn_mod as nn
import numpy as np
import pickle
import csv

'''
This code creates glove embeddings of text data. 
For more information about glove, see https://nlp.stanford.edu/projects/glove/
'''


# choose glove embedding dimensionality (50,100,200,300)
glove_dims = "50"

# paths for reading files
data_path = "/Users/punit/stat-nlp-book/data/nn/"
glove_path = 'glove/glove.6B.'+glove_dims+'d.txt'

# output paths
embed_path = 'obj/emb_full_'+glove_dims+'.pkl'

# read glove embeddings
df = pd.read_csv(glove_path, 
	header=None, 
	quoting=csv.QUOTE_NONE,
	na_values=None,
	keep_default_na=False,
	delim_whitespace=True)

# create global glove dictionary of terms
words = df[0]
vocab = {}
for i,w in enumerate(words):
	vocab[w] = i

# embeddings as ndarray
glove_embeddings = df[df.columns[1:]].as_matrix()
# size of embedding space
glove_size = glove_embeddings.shape[1]

# read data and create vocabulary
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
data_full = data_train + data_dev    ################### include dev data when submitting final model!
_, _, rel_vocab = nn.pipeline(data_full)

# grow embeddings matrix 
embeddings = np.zeros([len(rel_vocab), glove_size])
oov_emb = np.random.normal(scale=0.1, size=[1,glove_size])
for k,v in rel_vocab.items():
	if k in vocab:
		embeddings[v,:] = glove_embeddings[vocab[k],:]
	else:
		embeddings[v,:] = oov_emb
pad_emb = np.random.normal(scale=0.1, size=[1,glove_size])
embeddings[0,:] = pad_emb

# save embeddings as pickle objects
with open(embed_path, 'wb') as handle:
	pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)











