# -*- coding: utf-8 -*-
"""
Created on Thu Jul  19 15:02:01 2018

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

# Set maximum number of epochs
max_epoch = 5

#Split sccording to lines
rel = st.splitlines()

#Selectively parse the first and third columns and store them in a list
#Add additional text at the beginning and end of sentences
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

# Define the indicator function required in training
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
    

def SGD(Xttrans,train_labels,theta_init,max_epoch):
    
    '''
    Stochastic Gradient Descent (SGD) for training parameter matrix
    using negative log likelihood as the error metric.

    Parameters:
    ----------
    1. Xttrans: (np.matrix) Numerical feature matrix for training.
    2. train_labels: (list) List of class labels of the training data.
    3. theta_init: (np.matrix) Initialization of parameter matrix.
    4. max_epoch: (int) Number of epochs.

    Returns:
    ----------
    1. theta: (np.matrix) Trained parameter matrix.
    2. grad: (np.matrix) Matrix of gradient of negative log likelihood with
                respect to the parameter matrix theta.
    3. nlltrain: (list) List of negative log likelihood on the training data after every epoch.
    4. nllvalid: (list) List of negative log likelihood on the validation data after every epoch.
    '''
    
    # Initialize all result parameters
    theta = theta_init
    nllvalid = []
    nlltrain = []
    epoch = 0
    
    #SGD loop
    while epoch < max_epoch:
        
        #Set negative log likelihood to zero before every epoch
        nll=0
        
        #Initialize gradient matrix to matrix of zeros
        grad = np.matrix(np.zeros([3*M-2,K]))
        
        for i,example in enumerate(Xttrans): #Over N
            denom=0
            for z in range(K):
                term=0
                for exam in example:
                    term=term+(theta[exam,z])
                denom = denom + np.exp(term + theta [-1,z])

            for j,k in enumerate(range(K)): #Over K
                term2 = 0
                for exam in example:
                    term2=term2+(theta[exam,k])
                term2=term2+theta[-1,k]
                num=np.exp(term2)
                I=Indi(train_labels[i],k)
                ans=-(I - num/denom)
                for exam in example:
                    grad[exam,k] = ans
                grad[-1,k] = ans

            for exam in example:
                theta[exam] = theta[exam] - 0.5 * grad[exam]
            theta[-1,:] = theta[-1,:] - 0.5 * grad[-1,:]
            theta_init=theta
            
        # Keep track of negative log likelihood on training data
        nll=0
        for i,example in enumerate(Xttrans):
            denom=0
            num=0
            for z in range(K):
                term=0
                for exam in example:
                    term=term+(theta[exam,z])
                denom = denom + np.exp(term + theta [-1,z])

            for j,k in enumerate(range(K)):
                term2 = 0
                for exam in example:
                    term2=term2+(theta[exam,k])
                term2=term2+theta[-1,k]
                num=np.exp(term2)
                nll=nll-Indi(train_labels[i],k)/N*np.log(num/denom)
        nlltrain.append(nll)

        # Keep track of negative log likelihood on validation data
        nll=0
        for i,example in enumerate(Xvtrans):
            denom=0
            num=0
            for z in range(K):
                term=0
                for exam in example:
                    term=term+(theta[exam,z])
                denom = denom + np.exp(term + theta [-1,z])

            for j,k in enumerate(range(K)):
                term2 = 0
                for exam in example:
                    term2=term2+(theta[exam,k])
                term2=term2+theta[-1,k]
                num=np.exp(term2)
                nll=nll-Indi(valid_labels[i],k)/Nvalid*np.log(num/denom)
        nllvalid.append(nll)
        epoch = epoch + 1
        print ('Completed epoch: {}'.format(epoch))
    return theta,grad,nlltrain,nllvalid

theta_init=np.matrix(np.zeros([3*M-2,K]))

theta,grad,nlltrain,nllvalid=SGD(Xttrans,train_labels,theta_init,max_epoch)

# Train predictions
predtrain=[]
errtrain = 0
for i,example in enumerate(Xttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        term=0
        for exam in example:
            term = term + theta[exam,k]
        num.append(term+theta[-1,k])
    predtrain.append(np.argmax(num))
    if predtrain[i] != train_labels[i]:
        errtrain= errtrain + 1

# Test predictions
predtest=[]
errtest = 0
for i,example in enumerate(Xtesttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        term=0
        for exam in example:
            term = term + theta[exam,k]
        num.append(term+theta[-1,k])
    predtest.append(np.argmax(num))
    if predtest[i] != test_labels[i]:
        errtest = errtest + 1

print (errtrain/N,' and ',errtest/Ntest)

# Calculate error metrics
precision, recall, fscore, class_count = score(test_labels, predtest, labels=np.unique(test_labels))

print ('Confusion matrix:')
print (confusion_matrix(test_labels, predtest, labels=np.unique(test_labels)))

print ('F1-Scores:')
print (fscore)
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
