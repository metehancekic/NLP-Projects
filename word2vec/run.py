# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 00:46:42 2018

@author: mcekic
"""


import numpy as np
from random import sample

w_s=3
hidden_neuron_num=300
eta=0.01  # learning rate
epoch_max=1

def sigmoid(x):
    e=1/(1+np.exp(-x))
    return e

def softmax(x):
    e=np.exp(x-np.max(x))
    return e/e.sum()

def read_dict(file_name):
    
    file = open(file_name,"r")
    dict_size =0
    vocab_dict={}
    
    for line in file:
        word=line.replace('\n','')
        vocab_dict[word]=dict_size+1
        dict_size = dict_size+1
        
    return dict_size, vocab_dict

def read_corpus(dict_size,vocab_dict):
    word_freq=np.zeros((dict_size,), dtype=int)
    counter=0
    corpus_words=[]
    char_list=[]
    negative_words=[]
    corpus=open('text8')
    for line in corpus.read():    
        if (line==' '):
            counter+=1
            corpus_words.append("".join(char_list))  
            char_list=[]
            try:
                word_freq[vocab_dict[corpus_words[counter-1]]-1]+=1
                negative_words.append(corpus_words[counter-1])
            except KeyError:
                del corpus_words[counter-1]
                counter-=1
                continue    
        else:
            char_list.append(line)
    return negative_words,corpus_words,word_freq,counter

def forward_propagate(word,w1,w2):
    hidden=w1[word-1,:]
    outt=hidden.dot(w2)
    outputs=softmax(outt)
    return hidden,outputs

def negative_sampling(negative_words,vocab_dict):
    y=sample(negative_words, k=5)
    z=np.array([],dtype=np.int32)
    for i in range(5):
        z=np.append(z,vocab_dict[y[i]])
    return z

def back_prop(w1,w2,eta,negative_samples,x,h):
    EH=np.zeros(300)
    EI=0
    counterr=0
    t=np.ones(len(negative_samples))
    t[:5]=0
    for k in negative_samples: 
        EI=sigmoid((w2[:,k-1]).dot(w1[x-1,:]))-t[counterr]
        EH+=EI*w2[:,k-1]
        w2[:,k-1]-=eta*h*EI
        counterr+=1
        w1[x-1,:]-=eta*EH
    return w1,w2

def vector_saver(w1,vocab_dict,epoch):
    filename='vectors'+str(epoch)+'.txt'
    f = open(filename, 'w+')
    for word in vocab_dict:
        line_string = word+" "+' '.join(map(str,w1[vocab_dict[word]-1]))
        f.write(line_string+'\n')
    f.close()
    
def main():
    dict_size,vocab_dict=read_dict('vocab.txt')
    negative_words,corpus_words,word_freq,counter=read_corpus(dict_size,vocab_dict)       
    w1=np.random.normal(0,0.04,[dict_size,hidden_neuron_num])
    w2=np.random.normal(0,0.04,[hidden_neuron_num,dict_size])
    
    for epoch in range(epoch_max):
        for i in range(w_s,len(corpus_words)-w_s):
            x=0
            y=np.zeros(2*w_s+1)
            x=vocab_dict[corpus_words[i]]
            for indd in range(-w_s,w_s):
                y[indd+w_s]=vocab_dict[corpus_words[i+indd]]
            y=np.delete(y,w_s)
            y=y.astype(int)
            h,outputs=forward_propagate(x,w1,w2)
            negative_samples=negative_sampling(negative_words,vocab_dict)
            y_samples=np.append(negative_samples,y)
            if i%100000==0:
                print('%d' %i)
            w1,w2=back_prop(w1,w2,eta,y_samples,x,h)
        vector_saver(w1,vocab_dict,epoch)
        
if __name__=='__main__':
    main()


