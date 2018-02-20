# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 07:55:38 2018

@author: mcekic
"""

import numpy as np

w_s=3
vector_size=300
eta=0.1
epoch_max=1

def weighting_function(X):
    alpha=3/4
    x_max=40
    gridd=(X<x_max).astype(int)
    res=(np.multiply(X,gridd)/x_max)**alpha
    res=res+1-gridd
    return res


def read_dict(file_name):
    
    file = open(file_name,"r")
    dict_size =0
    vocab_dict={}
    
    for line in file:
        word=line.replace('\n','')
        vocab_dict[word]=dict_size+1
        dict_size = dict_size+1
        
    return dict_size, vocab_dict

def read_corpus(dict_size,vocab_dict,w_s):
    corpus_words=[]
    char_list=[]
    counter=0
    corpus=open('text8')
    X=np.zeros((dict_size,dict_size))
    for line in corpus.read():    
        if (line==' '):
            counter+=1
            corpus_words.append("".join(char_list))  
            char_list=[]
            try:
                
                if counter>w_s:
                    for i in range(-w_s,-1):
                        X[vocab_dict[corpus_words[counter-1]]-1,vocab_dict[corpus_words[counter+i-1]]-1]+=-1.0/float(i)
                        X[vocab_dict[corpus_words[counter+i-1]]-1,vocab_dict[corpus_words[counter-1]]-1]+=-1.0/float(i)
                else:
                    X[vocab_dict[corpus_words[counter-1]]-1,0]=0    
            except KeyError:
                del corpus_words[counter-1]
                counter-=1
                continue    
        else:
            char_list.append(line)
    return X,corpus_words

def initialize_weights(variance,a,b):
     w1=np.random.normal(0,variance,[a,b])
     w2=np.random.normal(0,variance,[b,a])
     b1=np.random.normal(0,variance,[a])
     b2=np.random.normal(0,variance,[a])
     return w1,w2,b1,b2
 
def cost_calc(w1,w2,b1,b2,X,dict_size,vector_size):
    A=np.matmul(w1,w2)+np.tile(b1,(dict_size,1))+np.transpose(np.tile(b2,(dict_size,1)))-np.log(X+1)
    B=weighting_function(X)
    J=np.sum(np.multiply(B,np.square(A)))   
    return J,A,B

def grad_back(w1,w2,A):
#    gradw1=weight*(w1.dot(w2)+b1+b2-np.log(X+1))*w2
#    gradw2=weight*(w1.dot(w2)+b1+b2-np.log(X+1))*w1
#    gradb1=weight*(w1.dot(w2)+b1+b2-np.log(X+1))
#    gradb2=weight*(w1.dot(w2)+b1+b2-np.log(X+1))
    gradw1=A*w2
    gradw2=A*w1

    
    return gradw1,gradw2
        
def vector_saver(w1,vocab_dict,epoch):
    filename='vectors'+str(epoch)+'.txt'
    f = open(filename, 'w+')
    for word in vocab_dict:
        line_string = word+" "+' '.join(map(str,w1[vocab_dict[word]-1]))
        f.write(line_string+'\n')
    f.close()

def main():
    dict_size,vocab_dict=read_dict('vocab.txt')
    w1,w2,b1,b2=initialize_weights(0.04,dict_size,vector_size)
    X,corpus_words=read_corpus(dict_size,vocab_dict,w_s)
    
    weight=weighting_function(X)
    
#    gradsq_W1 =1
#    gradsq_W2 =1
#    gradsq_b1 =1
#    gradsq_b2 =1
    
    for epoch in range(epoch_max):
        for i in range(w_s,len(corpus_words)-w_s):
            x=0
            y=np.zeros(2*w_s+1)
            x=vocab_dict[corpus_words[i]]-1
        
            for indd in range(-w_s,w_s):
                y[indd+w_s]=vocab_dict[corpus_words[i+indd]]-1
            y=np.delete(y,w_s)
            y=y.astype(int)
            for yc in y:
                A=weight[x,yc]*(w1[x,:].dot(w2[:,yc])+b1[x]+b2[yc]-np.log(X[x,yc]+1))
                g1,g2=grad_back(w1[x,:],w2[:,yc],A)
                w1[x,:]-=(eta*g1)#/np.sqrt(gradsq_W1)
                w2[:,yc]-=(eta*g2)#/np.sqrt(gradsq_W2)
                b1[x]-=(eta*A)#/np.sqrt(gradsq_b1)
                b2[yc]-=(eta*A)#/np.sqrt(gradsq_b2)
                #gradsq_W1 += np.square(g1)
                #gradsq_W2 += np.square(g2)
                #gradsq_b1 += g3 ** 2
                #gradsq_b2 += g4 ** 2
                
            if i%300000==0:
                J,A,B=cost_calc(w1,w2,b1,b2,X,dict_size,vector_size)
                per=100*float(i)/len(corpus_words)
                print('cost is %f at %f percentage at %d th epoch ' %(J,per,epoch)) 
        vector_saver(w1,vocab_dict,epoch)
        print('%d th epoch is done' %epoch)
    return w1,w2,b1,b2
        

if __name__=='__main__':
    w1,w2,b1,b2=main()
