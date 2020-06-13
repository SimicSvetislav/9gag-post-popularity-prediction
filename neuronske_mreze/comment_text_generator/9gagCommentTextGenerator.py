########################
#
# Potrebno je izmeniti putanje direktorijuma ako se kod pokrece na lokalnoj masini, jer je kod pisan i testiran na Google Colab-u
#
#
#
########################

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LambdaCallback
import random
import sys
import io

with open('/content/drive/My Drive/data/9gagRichText.txt', 'r') as rich_text:
        text = rich_text.read()

#unique characters
unique_characters = sorted(set(text))
print(len(unique_characters))

#map char to index & index to char
char2index = {u: i for i, u in enumerate(unique_characters)}
index2char = np.array(unique_characters)
text_as_int = np.array([char2index[c] for c in text])

seq_len = 50
examples_per_epoch = len(text)

#create training/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

#for item in sequences.take(5):
#  print(repr(''.join(index2char[item.numpy()])))

def creat_input_target_text(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(creat_input_target_text)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)
#dataset

vocab_size = len(unique_characters)
embedding_dim = 256
rnn_units = 512
batch_size = BATCH_SIZE

#MODEL
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))

model.add(GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
#model.add(Dropout(0.2))

model.add(GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
#model.add(Dropout(0.2))

model.add(GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
#model.add(Dropout(0.2))

model.add(Dense(vocab_size, activation="softmax"))

opt = Adam(lr=0.001, decay=1e-5)
model.compile(loss='sparse_categorical_crossentropy',
                optimizer = opt,
                metrics=['accuracy'])

checkpoint_dir = '/content/drive/My Drive/data/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckp_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 100
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


######
#model sa novim shape-om ulazen dimenzije 
######

model2 = Sequential()

model2.add(Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))

model2.add(GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
#model.add(Dropout(0.2))

model2.add(Dense(vocab_size, activation="softmax"))

model2.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model2.build(tf.TensorShape([1, None]))
model2.summary()


#######
#izgenerisi tekst na osnovu pocetnog ulaza
#######

def generate_text(model, start_string):
  num_generate = 400

  input_eval = [char2index[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  #print(input_eval.shape)

  text_generated = []

  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(index2char[predicted_id])

  return(start_string + ''.join(text_generated)) 

print(generate_text(model2, start_string=u"What is this"))

