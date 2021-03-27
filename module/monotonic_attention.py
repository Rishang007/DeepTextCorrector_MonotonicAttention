# Importing required libraries

import tensorflow as tf
import pandas as pd
import numpy as np
import regex as re
import pickle
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import initializers, regularizers, constraints

# Encoder 

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length, embedding_matrix_per):
        super().__init__()

        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.embedding_matrix_per = embedding_matrix_per

        self.embedd = Embedding(input_dim = self.inp_vocab_size, output_dim = self.embedding_size, input_length=self.input_length,
                                weights = [self.embedding_matrix_per], mask_zero=True)
        
        self.encod_lstm = LSTM(units = self.lstm_size, return_sequences=True, return_state=True, 
                               name="Encoder_Attention", activation = 'tanh',recurrent_activation ='sigmoid', 
                               recurrent_dropout = 0, unroll=False, use_bias=True, 
                               kernel_regularizer= regularizers.l2(1e-6))

    def call(self,input_sequence, training=True):
        embeddings = self.embedd(input_sequence)

        encod_out, encod_h, encod_c = self.encod_lstm(embeddings)

        return encod_out, encod_h, encod_c

    
    def get_config(self):
        '''Config'''

        config = { 'inp_vocab_size': self.inp_vocab_size,
                  'embedding_size': self.embedding_size,
                  'lstm_size': self.lstm_size,
                  'input_length': self.input_length,
                  'embedding_matrix_per': self.embedding_matrix_per
        }

        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Monotonic Attention required functions

def _attention_score(dec_ht,
                     enc_hs,
                     attention_type,
                     weightwa=None,
                     weightua=None,
                     weightva=None):
    if attention_type == 'bahdanau':
        score = weightva(tf.nn.tanh(weightwa(dec_ht) + weightua(enc_hs)))
        score = tf.squeeze(score, [2])
    elif attention_type == 'dot':
        score = tf.matmul(dec_ht, enc_hs, transpose_b=True)
        score = tf.squeeze(score, 1)
    elif attention_type == 'general':
        score = weightwa(enc_hs)
        score = tf.matmul(dec_ht, score, transpose_b=True)
        score = tf.squeeze(score, 1)
    elif attention_type == 'concat':
        dec_ht = tf.tile(dec_ht, [1, enc_hs.shape[1], 1])
        score = weightva(tf.nn.tanh(weightwa(tf.concat((dec_ht, enc_hs), axis=-1))))
        score = tf.squeeze(score, 2)
    return score


def _monotonic_attetion(probabilities, attention_prev, mode):

    if mode == 'hard':
        #Remove any probabilities before the index chosen last time step
        probabilities = probabilities*tf.cumsum(attention_prev, axis=1)
        attention = probabilities*tf.math.cumprod(1-probabilities, axis=1, exclusive=True)
    elif mode == 'recursive':
        batch_size = tf.shape(input=probabilities)[0]
        shifted_1mp_probabilities = tf.concat([tf.ones((batch_size, 1)),\
            1 - probabilities[:, :-1]], 1)
        attention = probabilities*tf.transpose(a=tf.scan(lambda x, yz: tf.reshape(yz[0]*x + yz[1],\
            (batch_size,)), [tf.transpose(a=shifted_1mp_probabilities),\
                tf.transpose(a=attention_prev)], tf.zeros((batch_size,))))
    elif mode == 'parallel':
        cumprod_1mp_probabilities = tf.exp(tf.cumsum(tf.math.log(tf.clip_by_value(1-probabilities,\
            1e-10, 1)), axis=1, exclusive=True))
        attention = probabilities*cumprod_1mp_probabilities*tf.cumsum(attention_prev/\
            tf.clip_by_value(cumprod_1mp_probabilities, 1e-10, 1.), axis=1)
    else:
        raise ValueError("Mode must be 'hard', 'parallel' or 'recursive' ")

    return attention


# Monotonic Attention Layer

class MonotonicBahaAtt(tf.keras.layers.Layer):
    def __init__(self, units,
                 mode,
                 return_aweights=False,
                 scaling_factor=None,
                 noise_std=0,
                 weights_initializer='he_normal',
                 bias_initializer='zeros',
                 **kwargs):
        
        if 'name' not in kwargs:
            kwargs['name'] = ""
            
        super(MonotonicBahaAtt, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.noise_std = noise_std
        self.weights_initializer = initializers.get(weights_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.weights_regularizer = regularizers.l2(1e-2)
    
    def build(self, input_shape):
        self._wa = layers.Dense(self.units, use_bias=False,\
            kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer,\
                kernel_regularizer= self.weights_regularizer, name=self.name+"Wa")
        
        self._ua = layers.Dense(self.units,\
            kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer,\
                kernel_regularizer= self.weights_regularizer, name=self.name+"Ua")
        
        self._va = layers.Dense(1, use_bias=False, kernel_initializer=self.weights_initializer,\
            kernel_regularizer= self.weights_regularizer,bias_initializer=self.bias_initializer, name=self.name+"Va")
        
        
    def call(self, decoder_hidden_state, encoder_outputs, prev_attention, training=True):

        encoder_outputs, decoder_hidden_state = tf.cast(encoder_outputs, tf.float32), \
            tf.cast(decoder_hidden_state, tf.float32)
        
        dec_hidden_with_time_axis = tf.expand_dims(decoder_hidden_state, 1)

        # score shape == (batch_size, max_length)
        score = _attention_score(dec_ht=dec_hidden_with_time_axis, enc_hs=encoder_outputs,\
                    attention_type='bahdanau', weightwa=self._wa,\
                        weightua=self._ua, weightva=self._va)
        
        if self.scaling_factor is not None:
            score = score/tf.sqrt(self.scaling_factor)


        if self.mode == 'hard':
            probabilities = tf.cast(score > 0, score.dtype)
        else:
            probabilities = tf.sigmoid(score)

        
        attention_weights = _monotonic_attetion(probabilities, prev_attention, self.mode)
        attention_weights = tf.expand_dims(attention_weights, 1)

        #context_vector shape (batch_size, hidden_size)
        context_vector = tf.matmul(attention_weights, encoder_outputs)
        context_vector = tf.squeeze(context_vector, 1, name="context_vector")

        return context_vector, tf.squeeze(attention_weights, 1, name='attention_weights')

    def get_config(self):
        config = {'units': self.units,
                  'mode': self.mode,
                  'scaling_factor': self.scaling_factor,
                  'noise_std': self.noise_std,
                  'weights_initializer': initializers.serialize(self.weights_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'weights_regularizer': self.weights_regularizer 
                  }

        base_config = super(MonotonicBahaAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# One step decoder

class OneStepDecoder(tf.keras.Model):

    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units,embedding_matrix):
        super().__init__()
        
        self.tar_vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.embedding_matrix = embedding_matrix

        self.decod_embedd = Embedding(input_dim = self.tar_vocab_size, output_dim = self.embedding_dim,input_length = self.input_length,
                                      weights = [self.embedding_matrix] , name="embedding_layer_onestepdecoder", mask_zero=True)
        
        self.decod_LSTM = LSTM(units = self.dec_units, return_state=True, activation = 'tanh', 
                               recurrent_activation ='sigmoid', recurrent_dropout = 0, unroll=False,use_bias=True,
                               kernel_regularizer= regularizers.l2(1e-6))
        
        self.MonotonicBahaAtt=MonotonicBahaAtt(self.att_units,
                 mode=self.score_fun,
                 return_aweights=False,
                 scaling_factor=None,
                 noise_std=0,
                 weights_initializer='he_normal',
                 bias_initializer='zeros',)

        self.dense = Dense(units=self.tar_vocab_size)

    def call(self,input_to_decoder, encoder_output, state_h,state_c, attention_weights):
        
        d_embedd = self.decod_embedd(input_to_decoder)

        context_vector,attention_weights = self.MonotonicBahaAtt(state_h,encoder_output, attention_weights)

        d_embedd = tf.concat([tf.expand_dims(context_vector,1), d_embedd], axis=-1)

        d_out, d_hidden, d_cell = self.decod_LSTM(d_embedd, initial_state=[state_h, state_c])

        output = self.dense(d_out)
        #print(d_out.shape)

        return output, d_hidden, d_cell, attention_weights, context_vector

    def get_config(self):

        config = {'tar_vocab_size': self.tar_vocab_size,
                  'embedding_dim': self.embedding_dim,
                  'input_length': self.input_length,
                  'dec_units': self.dec_units,
                  'score_fun': self.score_fun,
                  'att_units': self.att_units,
                  'embedding_matrix': self.embedding_matrix
                  }
        
        base_config = super(OneStepDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
       return cls(**config)


# Decoder

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, output_length, dec_units ,score_fun ,att_units, embedding_matrix):
        super().__init__()

        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.output_length = output_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.embedding_matrix = embedding_matrix

        self.onestep_decoder = OneStepDecoder(self.out_vocab_size, self.embedding_dim, self.output_length, self.dec_units,
                                              self.score_fun ,self.att_units, self.embedding_matrix)

        
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state, attention_weights ):

        all_outputs = tf.TensorArray(tf.float32, size=tf.shape(input_to_decoder)[1], name='output_arrays')  #size=input_to_decoder.shape[1]

        #shape = tf.shape(input_to_decoder)[1]
        #input_to_decoder.shape[1]  #range((tf.shape(input_to_decoder)[1]))
        #itr = tf.cast((tf.shape(input_to_decoder)[1]), dtype='int32')
        itr = input_to_decoder.get_shape().as_list()[1]
        #itr = tf.shape(input_to_decoder)[1]

        #_,itr = tf.shape(input_to_decoder)
        #itr = int(itr)

        for tstep in range(itr):    

            output, decoder_hidden_state, decoder_cell_state, attention_weights, context_vector = self.onestep_decoder(
            input_to_decoder[:, tstep:tstep+1], encoder_output, decoder_hidden_state, 
                                 decoder_cell_state, attention_weights)                      #default teacher forcing

            all_outputs = all_outputs.write(tstep, output)
        
        all_outputs = tf.transpose(all_outputs.stack(), [1,0,2])

        return all_outputs

    def get_config(self):
        config = {'out_vocab_size': self.out_vocab_size,
                  'embedding_dim': self.embedding_dim,
                  'output_length': self.output_length,
                  'dec_units': self.dec_units,
                  'score_fun': self.score_fun,
                  'att_units': self.att_units,
                  'embedding_matrix': self.embedding_matrix}

        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

# Attention_model

class attention_model(tf.keras.Model):
    def __init__(self,e_vocab_size, d_vocab_size, embedding_dim_e,embedding_dim_d, i_length, o_length, enc_units, 
                 dec_units ,score_fun ,att_units, embedding_matrix, embedding_matrix_per):
        super().__init__()

        self.e_vocab_size = e_vocab_size
        self.d_vocab_size = d_vocab_size
        self.embedding_dim_e = embedding_dim_e
        self.embedding_dim_d = embedding_dim_d
        self.i_length = i_length
        self.o_length = o_length
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.embedding_matrix = embedding_matrix
        self.embedding_matrix_per = embedding_matrix_per

        self.encoder = Encoder(self.e_vocab_size,self.embedding_dim_e,self.enc_units,self.i_length, self.embedding_matrix_per)
        self.decoder = Decoder(self.d_vocab_size,self.embedding_dim_d,self.o_length,self.dec_units,self.score_fun,self.att_units,self.embedding_matrix)

    def call(self, data, training=True):
        e_input,d_input = data[0], data[1]
        
        e_output, e_hidden, e_cell = self.encoder(e_input)
    
        d_hidden = e_hidden  #initial decoder state is equal to final encoder hidden state
        d_cell = e_cell

        attention_weights = np.zeros((512, 16), dtype='float32')
        attention_weights[:, 0] = 1
        
        final= self.decoder(d_input,e_output,d_hidden,d_cell, attention_weights)
        return final

    def get_config(self):
        
        config={'e_vocab_size': self.e_vocab_size,
                'd_vocab_size': self.d_vocab_size,
                'embedding_dim_e': self.embedding_dim_e,
                'embedding_dim_d': self.embedding_dim_d,
                'i_length': self.i_length,
                'o_length': self.o_length,
                'enc_units': self.enc_units,
                'dec_units': self.dec_units,
                'score_fun': self.score_fun,
                'att_units': self.att_units,
                'embedding_matrix': self.embedding_matrix,
                'embedding_matrix_per': self.embedding_matrix_per}
        
        base_config = super(attention_model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#References
# Monotonic Attention --> https://github.com/UdiBhaskar/TfKeras-Custom-Layers
# Attention model code --> Assignment done on attention