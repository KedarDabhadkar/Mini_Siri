# -*- coding: utf-8 -*-
"""
Created on Fri Aug  10 07:21:34 2018

@author: kdabhadk
"""

import time

# Start timer
start = time.time()

# import required modules
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# Import data
file ='ner_dataset.csv'
with codecs.open(file, "r",encoding='utf-8', errors='ignore') as doc:
    st = doc.read()

#Split sccording to lines
rel = st.splitlines()

# Set maximum number of epochs
max_epoch = 5

#Selectively parse the first and third columns and store them in a list
list_=[]
for i in rel[1:]:
    text = i.split(sep=',')
    if len(text) == 4:
        list_.append([text[1],text[3]])

#Store all the data in the numpy object all_data
#Cover all characters to lower case
all_data = np.char.lower(list_)

#Split training as the first 80% of all data
train = all_data[0:int(0.8*len(list_))]

# validation fraction is 10%
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]

# Test fraction is 10%
test = all_data[int(0.9*len(list_)):int(len(list_))]

def Indi (a , b):
    '''
    Returns 1 if a=b, else returns 0.

    Parametes
    ----------
    a,b : Objects

    Returns
    ----------
    Boolean, depending on similarity of objects a and b.
    '''
    if a == b :
        return 1
    else :
        return 0

# Making feature matrices for trainng, testing and validation
X=np.char.lower(train[:,0])
Xvalid=np.char.lower(valid[:,0])
Xtest=np.char.lower(test[:,0])

# Generate class vectors
class_=train[:,-1]
class_valid=valid[:,-1]
class_test=test[:,-1]

#Unique train labels
uniq_class=np.unique(all_data[:,-1])

#Get unique words
words=np.unique(all_data[:,0])

#Define numbers
M=len(words) + 1 #Number of rows of feature matrix for training
K=len(uniq_class) #Number of columns of feature matrix for training

N=len(class_) #Length of class vector for training
Ntest=len(Xtest) #Length of class vector for testing
Nvalid=len(Xvalid) #Length of class vector for validation

# Numerical array for training
Xttrans=[]
class_num=[]
for i,example in enumerate(X):
    Xttrans.append(words.tolist().index(example))
    class_num.append(uniq_class.tolist().index(class_[i]))

# Numerical array for valiation
Xvtrans=[]
classv_num=[]
for i,example in enumerate(Xvalid):
    Xvtrans.append(words.tolist().index(example))
    classv_num.append(uniq_class.tolist().index(class_valid[i]))

# Numerical array for test
Xtesttrans=[]
classtest_num=[]
for i,example in enumerate(Xtest):
    Xtesttrans.append(words.tolist().index(example))
    classtest_num.append(uniq_class.tolist().index(class_test[i]))

#Initialize a parameter matrix
theta_init=np.matrix(np.zeros([M,K]))

#Import logistic regression module
from sklearn.linear_model import LogisticRegression

#Define a logistic regression with L1 oenalty and balanced weights
log = LogisticRegression(penalty='l1',class_weight='balanced',verbose=1)

#Train on the training data
log.fit(np.array(Xttrans).reshape(-1,1),class_)

#Predictions on the test data
predicted = log.predict(np.array(Xtesttrans).reshape(-1,1))

#Calculate error metrics
precision, recall, fscore, class_count = score(class_test, predicted, labels=uniq_class)

print ('Confusion matrix:')
print (confusion_matrix(class_test, predicted, labels=uniq_class))

print ('F1-Scores:')
print (fscore)


print ('\n Classification metrics: \n')
for i in range(uniq_class.shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nAverage F1 MACRO: {0:.5f}'.format(np.sum(np.array(class_count)*np.array(fscore))/sum(class_count)))

end = time.time()

print ('Total run time:{}'.format(end-start))
