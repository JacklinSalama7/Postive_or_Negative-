# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings
import nltk 
import re 
import xlrd
import math
import os
import random
import sys
import string
import gensim 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from gensim.models import Word2Vec , keyedvectors
from nltk.corpus import wordnet as wn
import xlwt 
from xlwt import Workbook 
import xlsxwriter
workbook = xlsxwriter.Workbook('output.xlsx') 
workbook = xlsxwriter.Workbook('upper.xlsx') 
workbook = xlsxwriter.Workbook('tw.xlsx') 
workbook = xlsxwriter.Workbook('word2vec.xlsx') 
worksheet = workbook.add_worksheet() 
def uppercase(data):
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 3')
    col_num = 1
    for i in range(len(data)):
        str=  data.Text[i] 
        count=0
        for j in str:
            if j.isupper():
                count+=1
            else:
                continue
    
        print((count/len(str))*100)
        up=(count/len(str))*100
       # cas = ' '.join(map(str, up)) 
        #sheet1.write(col_num, 1,up) 
        #col_num += 1
        #wb.save('upper.xlsx')
       

def post(data):
    for i in range(len(data)):
        str=  data.Text[i]  
        str= str.strip()
        words=list(str.split(' '))  
        pos_all = dict()
        for w in words:
            pos_l = set()
            for tmp in wn.synsets(w):
                if tmp.name().split('.')[0] == w:
                    pos_l.add(tmp.pos())
            pos_all[w] = pos_l
        print (pos_all)
  


def softmax(x): 
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x)) 
	return e_x / e_x.sum() 

class word2vec(object): 
	def __init__(self): 
		self.N = 10
		self.X_train = [] 
		self.y_train = [] 
		self.window_size = 2
		self.alpha = 0.001
		self.words = [] 
		self.word_index = {} 

	def initialize(self,V,data): 
		self.V = V 
		self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N)) 
		self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V)) 
		
		self.words = data 
		for i in range(len(data)): 
			self.word_index[data[i]] = i 

	
	def feed_forward(self,X): 
		self.h = np.dot(self.W.T,X).reshape(self.N,1) 
		self.u = np.dot(self.W1.T,self.h) 
		#print(self.u) 
		self.y = softmax(self.u) 
		return self.y 
		
	def backpropagate(self,x,t): 
		e = self.y - np.asarray(t).reshape(self.V,1) 
		# e.shape is V x 1 
		dLdW1 = np.dot(self.h,e.T) 
		X = np.array(x).reshape(self.V,1) 
		dLdW = np.dot(X, np.dot(self.W1,e).T) 
		self.W1 = self.W1 - self.alpha*dLdW1 
		self.W = self.W - self.alpha*dLdW 
		
	def train(self,epochs): 
		for x in range(1,epochs):		 
			self.loss = 0
			for j in range(len(self.X_train)): 
				self.feed_forward(self.X_train[j]) 
				self.backpropagate(self.X_train[j],self.y_train[j]) 
				C = 0
				for m in range(self.V): 
					if(self.y_train[j][m]): 
						self.loss += -1*self.u[m][0] 
						C += 1
				self.loss += C*np.log(np.sum(np.exp(self.u))) 
			print(self.loss," : ",end='') 
			self.alpha *= 1/( (1+self.alpha*x) ) 
			
	def predict(self,word,number_of_predictions): 
		if word in self.words: 
			index = self.word_index[word] 
			X = [0 for i in range(self.V)] 
			X[index] = 1
			prediction = self.feed_forward(X) 
			output = {} 
			for i in range(self.V): 
				output[prediction[i][0]] = i 
			
			top_context_words = [] 
			for k in sorted(output,reverse=True): 
				top_context_words.append(self.words[output[k]]) 
				if(len(top_context_words)>=number_of_predictions): 
					break
	
			return top_context_words 
		#else: 
    
			#print("Word not found in dicitonary") 
def preprocessing(corpus): 
	stop_words = set(stopwords.words('english'))	 
	training_data = [] 
	sentences = corpus.split(".") 
	for i in range(len(sentences)): 
		sentences[i] = sentences[i].strip() 
		sentence = sentences[i].split() 
		x = [word.strip(string.punctuation) for word in sentence 
									if word not in stop_words] 
		x = [word.lower() for word in x] 
		training_data.append(x) 
	return training_data 
	

def prepare_data_for_training(sentences,w2v): 
	data = {} 
	for sentence in sentences: 
		for word in sentence: 
			if word not in data: 
				data[word] = 1
			else: 
				data[word] += 1
	V = len(data) 
	data = sorted(list(data.keys())) 
	vocab = {} 
	for i in range(len(data)): 
		vocab[data[i]] = i 
	
	#for i in range(len(words)): 
	for sentence in sentences: 
		for i in range(len(sentence)): 
			center_word = [0 for x in range(V)] 
			center_word[vocab[sentence[i]]] = 1
			context = [0 for x in range(V)] 
			
			for j in range(i-w2v.window_size,i+w2v.window_size): 
				if i!=j and j>=0 and j<len(sentence): 
					context[vocab[sentence[j]]] += 1
			w2v.X_train.append(center_word) 
			w2v.y_train.append(context) 
	w2v.initialize(V,data) 

	return w2v.X_train,w2v.y_train 



def word2vector(data):
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 3')
    col_num = 1
    for i in range(len(data)):
        str1=  data.Text[i] 
        corpus = "" 
        corpus += str1
        epochs = 2

        training_data = preprocessing(corpus) 
        w2v = word2vec() 

        prepare_data_for_training(training_data,w2v) 
        w2v.train(epochs) 

        print(w2v.predict("around",3))
        
        #sheet1.write(col_num, 2,w2v.predict("around",3)) 
       # col_num += 1
      #  wb.save('word2vec.xlsx')
        

def retweet(data):
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 2')
    col_num = 1
    for i in range(len(data)):
        ch =False
        str1=  data.Text[i] 
        str1= str1.strip()
        words=list(str1.split(' '))  
        for j in range(len(words)):
            words[j]= words[j].lower()
            if words[j]=="tweet":
                ch=True
                break
            else:
                continue
        print(ch)
        #sheet1.write(col_num, 2,ch ) 
        #col_num += 1
        #wb.save('tw.xlsx')

def postag(data):
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')
    col_num = 1
    for i in range(399,700):
        str1=  data.Text[i] 
        text =str1
        text = nltk.word_tokenize(text)
        result = nltk.pos_tag(text)
        #print(result)
        listToStr = ' '.join(map(str, result)) 
        print (listToStr)

      #  sheet1.write(col_num, 0,listToStr ) 
       # col_num += 1
       # wb.save('output.xlsx')
   
    
    

data = pd.read_excel (r'task2.xlsx')
uppercase(data)
postag(data)
retweet(data)
word2vector(data)


   












    
 
    















