#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:51:14 2017

@author: purna
"""
import numpy as np
from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
#from os.path import isfile,join
#import codecs
import re

def GloveModel():
    try:
        wordvec = np.load('wordVector.npy')
        words = np.load('words.npy')
    except:
        glove = open('glove.6B.50d.txt','r');
        wordvec = [];
        words = [];
        
        for line in glove:
            word_vec = [];
            word_vec = line.split()
            words.append(word_vec[0]);
            wordvec.append(word_vec[1:]);
    #print(words)
    ##np.save('wordvec',wordvec)
    return wordvec,words;

def cleanWords(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
    
def AverageWords():
    
    pos_items = listdir('pos/');
    neg_items = listdir('neg/')
    processed_pos_paths = []
    processed_neg_paths = []
    for item in pos_items :
        processed_pos_paths.append('pos/' + item)
    for item in neg_items :
        processed_neg_paths.append('neg/' + item)
    print(len(processed_pos_paths))
    print(len(processed_neg_paths))
    numwords = []
    for file_path in processed_pos_paths :
        with open(file_path , 'r',encoding = 'latin-1') as pf:
            line = pf.read()
            pf.close()
            line = cleanWords((line)) 
            numwords.append(len(line.split()))
            
    
    for file_path in processed_neg_paths :
        with open(file_path , 'r',encoding = 'latin-1') as nf:
            line = nf.readline()
            line = cleanWords(line)
            numwords.append(len(line.split()))
    
    tot = 0 
    for num in numwords :
        tot += num
    
    average_words = int(tot/1403)
    return average_words,processed_pos_paths,processed_neg_paths,tot;

def createTensor(numWords, maxSeqLength, positiveFiles, negativeFiles):
    
    numReviews = 1403
    (wordsList) = list(np.load('words.npy'))
    try:
        ids = np.load('idsMatrix.npy')
        output = np.load('output.npy')
    except:
        ids = np.zeros((numReviews, maxSeqLength))
        output = np.zeros((numReviews,2))  #positive = [1,0], negative = [0,1]
        fileCounter = 0
        for pf in positiveFiles:
            with open(pf, "r",encoding = 'latin-1') as f:
                output[fileCounter] = [1, 0]
                indexCounter = 0
                line=f.readline()
                cleanedLine = cleanWords(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        ids[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1 
        
        for nf in negativeFiles:
            with open(nf, "r",encoding = 'latin-1') as f:
                output[fileCounter] = [0, 1]
                indexCounter = 0
                line=f.readline()
                cleanedLine = cleanWords(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        ids[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1
        np.save('idsMatrix', ids)
        np.save('output', output)
    return ids, output

def EmbededMatrix(average_words):
    #words = np.load('words.npy')
    try:
        tensor = np.load('tensor_lstm.npy')
    except:
        ids = (np.load('idsMatrix.npy'))
        #print(ids)
        vectors = list(np.load('wordVectors.npy')) #to be made
        print(type(vectors))
        tensor = np.zeros([1403,645,50])
        print(tensor.shape)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
               tensor[i][j] = vectors[int(ids[i][j])]
        np.save('tensor_lstm',tensor)     
        
    return tensor

def share_train_test(words_embeded,output):
    random = 0.2
    x_test = np.zeros([int(1403*0.2),645,50])
    y_test = np.zeros([int(1403*0.2),2])
    x_train = words_embeded
    y_train = output
    for i in range(int(1403*0.2)):
        for j in range(int(1403*0.2)):
            x_test[i][j] = words_embeded[i][j]
            y_test[i] = output[i]
    print(x_test.shape,y_test.shape)
    return x_train,y_train,x_test,y_test
    
    
def LSTM_model(x_train,y_train,x_test,y_test):
    try:
        model = load_model('pretrain.h5')
    except:
        model = Sequential()
        model.add(LSTM(128, input_shape=(645,50)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        output = model.fit(x_train, y_train,  epochs=20, batch_size=64, verbose=1, validation_data=(x_test, y_test))
        model.save('pretrain.h5')
    print(x_test)
    print(model.predict(x_test))
    return model

if __name__ == "__main__":
    print("uploding the glove model.....");
    #wordvec,words = GloveModel();
    print("analysing for preprocessing ......");
    average_words,processed_pos_paths,processed_neg_paths,tot =  AverageWords()
    print(average_words,tot)
    print("tensor getting prepared for input of size 1403*645*50......")
    ids,output = createTensor(tot,average_words,processed_pos_paths,processed_neg_paths)
    word_embeded = EmbededMatrix(average_words);
    print("embeded matrix for input has been created ......")
    x_train,y_train,x_test,y_test = share_train_test(word_embeded,output)
    model = LSTM_model(x_train,y_train,x_test,y_test)
    #WordsDataSet();
    #ids = np.load('idsMatrix.npy')
    #print(ids)