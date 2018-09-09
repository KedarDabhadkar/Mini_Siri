# -*- coding: utf-8 -*-
"""
Created on Sat Jul  28 08:38:49 2018

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
from sklearn.linear_model import LogisticRegression


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
for i,datapoint in enumerate(rel[1:]):
    text = datapoint.split(sep=',')
    if len(text) == 4:
        if text[0][0:8]=='Sentence':
            list_.append(['EOS','EOS'])
            list_.append(['BOS','BOS'])
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

def generate(trainmodify):
    
    '''
    Function to generate the required sequential feature matrix.
    
    Parameters
    ----------
    trainmodify: List of data that has to be converted.
    
    Returns
    ----------
    Xttrans: Transformed array of data
    label: List of correspondong labels
    '''
    
    Xttrans=[]
    label=[]

    for i in range(1,len(trainmodify)-1):
        ap=[]
        firstword=trainmodify[i-1,0]
        wordmid=trainmodify[i,0]
        thirdword=trainmodify[i+1,0]

        if wordmid != 'eos':
            if wordmid != 'bos':
                ap.append(words.tolist().index(firstword))
                ap.append(M-1+words.tolist().index(wordmid))
                ap.append(2*M-2+words.tolist().index(thirdword))

                label.append(uniq_class.tolist().index(trainmodify[i,1]))
        if len(ap) != 0:
            Xttrans.append(ap)
    return Xttrans,label


# Numerical array for training
Xttrans,train_labels=generate(train)

# Numerical array for valiation
Xvtrans,valid_labels=generate(valid)

# Numerical array for test
Xtesttrans,test_labels=generate(test)

#Define a logistic regression with L1 oenalty and balanced weights
log = LogisticRegression(penalty='l1',class_weight='balanced')

#Train on the training data
log.fit(np.array(Xttrans).reshape(-1,3),train_labels)

#Predictions on the test data
predicted = log.predict(np.array(Xtesttrans).reshape(-1,3))

#Calculate error metrics
print ('Confusion matrix:')
print(confusion_matrix(test_labels, predicted, labels=np.unique(test_labels)))

precision, recall, fscore, class_count = score(test_labels, predicted, labels=np.unique(test_labels))

print ('F1-Scores:')
print(fscore)

print ('\n Classification metrics: \n')
for i in range(np.unique(test_labels).shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nAverage F1 MACRO: {0:.5f}'.format(np.sum(np.array(class_count)*np.array(fscore))/sum(class_count)))

end = time.time()

print ('Total run time:{}'.format(end-start))
