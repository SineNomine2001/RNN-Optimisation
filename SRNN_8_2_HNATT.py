'''
Author: Zeping Yu
Sliced Recurrent Neural Network (SRNN). 
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
This work is accepted by COLING 2018.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.
'''

import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense, Bidirectional
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import codecs

#https://github.com/minqi/hnatt
class Attention(Layer):  
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context', 
									   shape=(input_shape[-1], 1),
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)
		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')
		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
		return weighted_sum

	def compute_output_shape(self, input_shape):
		print(input_shape)
		return (input_shape[0], input_shape[-1])

#load data
df = pd.read_csv("yelp_2014.csv")
#df = df.sample(5000)

Y = df.stars.values-1
Y = to_categorical(Y,num_classes=5)
X = df.text.values

#set hyper parameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT=0.1
NUM_FILTERS = 50
MAX_LEN = 512
Batch_size = 100
EPOCHS = 5

#shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(2018)
np.random.shuffle(indices)
X=X[indices]
Y=Y[indices]

#training set, validation set and testing set
nb_validation_samples_val = int((VALIDATION_SPLIT + TEST_SPLIT) * X.shape[0])
nb_validation_samples_test = int(TEST_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val =  X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val =  Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

#use tokenizer to build vocab
tokenizer1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer1.fit_on_texts(df.text)
vocab = tokenizer1.word_index

x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_test_word_ids = tokenizer1.texts_to_sequences(x_test)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)

#pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)

#slice sequences into many subsequences
x_test_padded_seqs_split=[]
for i in range(x_test_padded_seqs.shape[0]):
    split1=np.split(x_test_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_test_padded_seqs_split.append(a)

x_val_padded_seqs_split=[]
for i in range(x_val_padded_seqs.shape[0]):
    split1=np.split(x_val_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_val_padded_seqs_split.append(a)
   
x_train_padded_seqs_split=[]
for i in range(x_train_padded_seqs.shape[0]):
    split1=np.split(x_train_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_train_padded_seqs_split.append(a)

#load pre-trained GloVe word embeddings
print("Using GloVe embeddings")
glove_path = 'glove.6B.200d.txt'
embeddings_index = {}
f = codecs.open(glove_path, encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#use pre-trained GloVe word embeddings to initialize the embedding layer
embedding_matrix = np.random.random((MAX_NUM_WORDS + 1, EMBEDDING_DIM))
for word, i in vocab.items():
    if i<MAX_NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be random initialized.
            embedding_matrix[i] = embedding_vector
            
embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=int(MAX_LEN/64),
                            trainable=True)

ATT_SIZE = 50

from keras import regularizers
l2_reg = regularizers.l2(1e-8)

#build model
print("Build Model")
input1 = Input(shape=(int(MAX_LEN/64),), dtype='int32')
embed1 = embedding_layer(input1)
gru1 = Bidirectional(GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=True))(embed1)
dense1 = TimeDistributed(Dense(ATT_SIZE))(gru1)# att
att1 = Attention(regularizer=l2_reg)(dense1)
Encoder1 = Model(input1, att1)

input2 = Input(shape=(8,int(MAX_LEN/64),), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = Bidirectional(GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=True))(embed2)
dense2 = TimeDistributed(Dense(ATT_SIZE))(gru2)# att
att2 = Attention(regularizer=l2_reg)(dense2)
Encoder2 = Model(input2, att2)

input3 = Input(shape=(8,8,int(MAX_LEN/64)), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = Bidirectional(GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=True))(embed3)
dense3 = TimeDistributed(Dense(ATT_SIZE))(gru3)# att
att3 = Attention(regularizer=l2_reg)(dense3)
preds = Dense(5, activation='softmax')(att3)
model = Model(input3, preds)

print(Encoder1.summary())
print(Encoder2.summary())
print(model.summary())

#use adam optimizer
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#save the best model on validation set
from keras.callbacks import ModelCheckpoint             
savebestmodel = 'save_model/SRNN(8,2)_yelp2013.h5'
checkpoint = ModelCheckpoint(savebestmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks=[checkpoint] 
             
model.fit(np.array(x_train_padded_seqs_split), y_train, 
          validation_data = (np.array(x_val_padded_seqs_split), y_val),
          epochs = EPOCHS, 
          batch_size = Batch_size,
          callbacks = callbacks,
          verbose = 2)

#use the best model to evaluate on test set
from keras.models import load_model
best_model= load_model(savebestmodel, custom_objects={'Attention': Attention})
print(best_model.evaluate(np.array(x_test_padded_seqs_split),y_test,batch_size=Batch_size, verbose=2))