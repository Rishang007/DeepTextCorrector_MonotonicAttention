import tensorflow as tf
import sys
import pickle
import numpy as np
import pandas as pd

import monotonic_attention

class model_att:

    def __init__(self, embedding_matrix_path, embedding_matrix_per_path, token_perturbation_path, token_correct_path, weights_path):

        try:
            self.embedding_matrix = pickle.load(open(embedding_matrix_path,'rb'))
        except:
            print("ERROR!! - Please enter a valid path for embedding_matrix")
        
        try:
            self.embedding_matrix_per = pickle.load(open(embedding_matrix_per_path,'rb'))
        except:
            print("ERROR!! - Please enter a valid path for embedding_matrix_per")
        
        try:
            self.token_perturbation = pickle.load(open(token_perturbation_path,'rb'))
        except:
            print("ERROR!! - Please enter a valid path for token_perturbation")
        
        try:
            self.token_correct = pickle.load(open(token_correct_path,'rb'))
        except:
            print("ERROR!! - Please enter a valid path for token_correct")

        v_perturbation = len(self.token_perturbation.word_index.keys()) +1
        v_correct=len(self.token_correct.word_index.keys())  + 1

        e_vocab_size = v_perturbation
        d_vocab_size = v_correct
        embedding_dim_e = 300
        embedding_dim_d = 300
        i_length = 16
        o_length = 16
        enc_units = 300
        dec_units = 300
        score_fun = 'parallel'
        att_units = 300

        self.attention = monotonic_attention.attention_model(e_vocab_size, d_vocab_size, embedding_dim_e,embedding_dim_d, 
            i_length, o_length, enc_units, dec_units ,score_fun ,att_units, self.embedding_matrix, self.embedding_matrix_per)


        self.attention.build((None,512,16))

        try:
            self.attention.load_weights(weights_path)
        except:
            print("ERROR!! - Please enter a valid path for model weights")


    def predict(self, inp):


        if isinstance(inp, list) is not True:
            return "ERROR!! - Please pass the input as a list of strings"

        encoder = self.attention.layers[0]
        decoder_layer = self.attention.layers[1]
        onestep = decoder_layer.layers[0]

        translation = np.empty((len(inp), 16), dtype='<U20')

        e_input=[]
        for i in inp:
            temp=[]
            for j in i.split():
                if self.token_perturbation.word_index.get(j) == None:
                    temp.append(0)
                else:
                    temp.append(self.token_perturbation.word_index.get(j))
            e_input.append(temp)
        
        e_input = tf.keras.preprocessing.sequence.pad_sequences(e_input, maxlen=16, padding='post')
        
        
        e_output, e_hidden, e_cell = encoder(e_input,0)
        
        d_hidden = e_hidden
        d_cell = e_cell                   #final encoder state is equal to initial decoder state
        x = self.token_correct.word_index['<start>']
        #initial decoder input is start
        d_input = tf.expand_dims([x for l in range(len(inp))],1)

        attention_weights = np.zeros((len(inp),16), dtype='float32')
        attention_weights[:,0] = 1

        for word in range(16):
            
            predicted, d_hidden, d_cell, attention_weights, context_vector = onestep(d_input, e_output, d_hidden, d_cell,
                                                                                        attention_weights)        
                    
            #making predictions
            predicted_word_index = tf.argmax(predicted, axis=1).numpy()  #it will give the highest probability word
        

            #stopping when reaching "<end>"
            #if self.token_correct.index_word[predicted_word_index] == '<end>':
                #break
            

            #appending the result with predicted words
            for i,j in enumerate(predicted_word_index):
                
                translation[i][word] = self.token_correct.index_word[j]
            
            # the predicted ID is fed back into the model
            d_input = tf.expand_dims(predicted_word_index, 1)
        
        translation = [list(i)[:list(i).index('<end>')] for i in translation]      #removing all the end
        translation = [' '.join(i) for i in translation]

        return translation



#References
# Monotonic Attention --> https://github.com/UdiBhaskar/TfKeras-Custom-Layers
# Attention model code --> Assignment done on attention