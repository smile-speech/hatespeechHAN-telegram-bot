from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



#import tensorflow
#https://github.com/tensorflow/tensorflow/issues/38503
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
experimental_run_tf_function=False
from tensorflow.compat.v1.keras.layers import CuDNNGRU
#from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from keras.engine.topology import Layer
#from tensorflow.compat.v1.keras.layers import CuDNNGRU
from keras.layers import Input, Embedding, Dense
from keras.layers import Lambda, Permute, RepeatVector, Multiply
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import GRU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from numpy import argmax



# pylint: disable=missing-docstring
import argparse

import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from os import environ
from importlib import reload

# Telegram Bot imports


class AttentionLayer(Layer):
    def __init__(self, attention_dim=100, **kwargs):
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], self.attention_dim),
                                 initializer='random_normal',trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(self.attention_dim, ),
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(self.attention_dim, 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
        
    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
        u_it = K.tanh(K.dot(x, self.W) + self.b)
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)
        
        return a_it
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
    
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        #config = {'attention_dim': self.attention_dim}
        config['attention_dim'] = self.attention_dim

        return config
    

def WeightedSum(attentions, representations):
    # from Shape(batch_size, len_units) to Shape(batch_size, rnn_dim * 2, len_units)
    repeated_attentions = RepeatVector(K.int_shape(representations)[-1])(attentions)
    # from Shape(batch_size, rnn_dim * 2, len_units) to Shape(batch_size, len_units, lstm_dim * 2)
    repeated_attentions = Permute([2, 1])(repeated_attentions)

    # compute representation as the weighted sum of representations
    aggregated_representation = Multiply()([representations, repeated_attentions])
    aggregated_representation = Lambda(lambda x: K.sum(x, axis=1))(aggregated_representation)

    return aggregated_representation




from keras.models import load_model


config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.compat.v1.Session(config=config)

K.set_session(session)






model= load_model('model1.hdf5',custom_objects={'AttentionLayer': AttentionLayer})
attention_extractor = load_model("model2.hdf5",custom_objects={'AttentionLayer': AttentionLayer},compile = False)
attention_extractor._make_predict_function()
model._make_predict_function()
#global graph
#graph = tf.compat.v1.get_default_graph()


import numpy as np
#텍스트 데이터는 전처리 과정을 통해 모델에 데이터를 정제하는 과정을 필요로함
#이러한 전처리 과정에 사용할 모듈
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize

import pickle


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_SENTENCES = 10
MAX_SENTENCE_LENGTH = 25

def doc2hierarchical(text,
                     max_sentences=MAX_SENTENCES,
                     max_sentence_length=MAX_SENTENCE_LENGTH):
    sentences = sent_tokenize(text)

    # loading
    #with open('tokenizer.pickle', 'rb') as handle:
     #   tokenizer = pickle.load(handle)
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)
    tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=max_sentence_length)

    pad_size = max_sentences - tokenized_sentences.shape[0]

    if pad_size <= 0:  # tokenized_sentences.shape[0] < max_sentences
        tokenized_sentences = tokenized_sentences[:max_sentences]
    else:
        tokenized_sentences = np.pad(
            tokenized_sentences, ((0, pad_size), (0, 0)),
            mode='constant', constant_values=0
        )
    
    return tokenized_sentences

def build_dataset(x_data, 
                  max_sentences=MAX_SENTENCES, 
                  max_sentence_length=MAX_SENTENCE_LENGTH,
                  tokenizer=tokenizer):
    
    nb_instances = len(x_data)
    
    
    X_data = np.zeros((nb_instances, max_sentences, max_sentence_length), dtype='int32')
    
    
    for i, review in enumerate(x_data):
        tokenized_sentences = doc2hierarchical(review)
            
        X_data[i] = tokenized_sentences[None, ...]
        

    
    return X_data




word_rev_index = {}
for word, i in tokenizer.word_index.items():
    word_rev_index[i] = word





def sentiment_analysis(review):
    pred_att=[]
    words = []
    tokenized_sentences = doc2hierarchical(review)
    #attention_extractor._make_predict_function()
    with session.as_default():
        with session.graph.as_default():
            pred_attention = attention_extractor.predict(np.asarray([tokenized_sentences]))[0][0]
            sent_attention = attention_extractor.predict(np.asarray([tokenized_sentences]))[1][0]
            pred_percent = model.predict(np.asarray([tokenized_sentences]))
            label = argmax(pred_percent)
            print(label)



    if label == 0:
        return 0,0,0,0,0



    elif label == 1:
        for sent_idx, sentence in enumerate(tokenized_sentences):
            if sentence[-1] == 0:
                continue
            for word_idx in range(MAX_SENTENCE_LENGTH):
                if sentence[word_idx] != 0:
                    words = [word_rev_index[word_id] for word_id in sentence[word_idx:]]
                    pred_att = pred_attention[sent_idx][-len(words):]
                #print(pred_att)
                #bot.loss_hist(pred_att)
                #pred_att = np.expand_dims(pred_att, axis=0)
                #print(pred_att)
                    break
    return 1, pred_attention, tokenized_sentences, sent_attention, word_rev_index
    #for i in range(len(pred_att)):
    #    a = pred_att[i]
        #print(a)
    #    bot.loss_hist.append(a)

    #for i in range(len(words)):
    #    a = words[i]
        #print(a)
    #    bot.word_list.append(a)

    #self.loss_hist = pred_att
    #self.word_list = words

    #for i in range(len(pred_att)):
    #   a = pred_att[i]
    #   #print(a)
    #   bot.loss_hist.append(a)

    # for i in range(len(words)):
    #     a = words[i]
    #     #print(a)
    #     bot.word_list.append(a)
        # fig, ax = plt.subplots(figsize=(len(words), 1))
        # plt.rc('xtick', labelsize=16)
        # midpoint = (max(pred_att[:, 0]) - min(pred_att[:, 0])) / 2
        # print(words)
        # heatmap = sn.heatmap(pred_att, xticklabels=words, yticklabels=False, square=True, linewidths=0.1, cmap='coolwarm', center=midpoint, vmin=0, vmax=1)
        # plt.xticks(rotation=45)
        # plt.show()
        # print(pred_att)
        #return pred_att
        
#sentiment_analysis("Delicious healthy food. The steak is amazing. Fish and pork are awesome too. Service is above and beyond. Not a bad thing to say about this place. Worth every penny!")

