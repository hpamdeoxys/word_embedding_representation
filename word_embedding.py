# keras has a buit in modeule for one_hot representation
from tensorflow.keras.preprocessing.text import one_hot

sent=[  'I love to play football',
     'I live in Bangalore',
     'Beautiful weather',
    'I live in USA',
     'Do not go out',
     'Trying word embeddings',
     'It started raining heavily']

voc_size=10000

onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)
# one hot representation of the sentences in the array "sent"

[[7941, 2350, 925, 5486, 6022], [7941, 3599, 8135, 4527], [6222, 628], [7941, 3599, 8135, 9696], [8707, 3467, 1415, 7878], [5730, 7677, 7668], [757, 8072, 6140, 154]]

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np

sent_length=10
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
# padding the vector so that each vector is of the same size
[[   0    0    0    0    0 7941 2350  925 5486 6022]
 [   0    0    0    0    0    0 7941 3599 8135 4527]
 [   0    0    0    0    0    0    0    0 6222  628]
 [   0    0    0    0    0    0 7941 3599 8135 9696]
 [   0    0    0    0    0    0 8707 3467 1415 7878]
 [   0    0    0    0    0    0    0 5730 7677 7668]
 [   0    0    0    0    0    0  757 8072 6140  154]]

model = Sequential()
model.add(Embedding(voc_size, 10, input_length=sent_length, name = "embed"))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embed (Embedding)            (None, 10, 10)            100000    
_________________________________________________________________
module_wrapper_4 (ModuleWrap (None, 100)               0         
_________________________________________________________________
module_wrapper_5 (ModuleWrap (None, 1)                 101       
=================================================================
Total params: 100,101
Trainable params: 100,101
Non-trainable params: 0'''

weights = model.get_layer('embed').get_weights()[0]


# From the above vector representation, we can print vectors for "Bangalore" and "USA", as both are geographical areas the vector should have been very close, but this is not
# true here, because the data set which I have taken is quite small. It should work fine for larger data sets

'''
#Bangalore
weights[4527]
array([ 0.03870297, -0.01750501,  0.01499713,  0.04920932,  0.04580829,
       -0.00054568, -0.00349169, -0.04235118,  0.0101148 ,  0.01633665],
      dtype=float32)
'''
'''
#USA
weights[9696]
array([0.01779553, 0.02971078, 0.04779202, 0.03299356, 0.04362563,
       0.04598795, 0.04074191, 0.01526001, 0.03561859, 0.04490686],
      dtype=float32)
      
'''
