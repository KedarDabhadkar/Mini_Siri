
# coding: utf-8

# In[1]:
import time
start = time.time()

import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score


# In[2]:


file ='ner_dataset.csv'
with codecs.open(file, "r",encoding='utf-8', errors='ignore') as doc:
    st = doc.read(999)

rel = st.splitlines()

max_epoch = 5

list_=[]
for i,datapoint in enumerate(rel[1:]):
    text = datapoint.split(sep=',')
    if len(text) == 4:
        if text[0][0:8]=='Sentence':
            list_.append(['EOS','EOS'])
            list_.append(['BOS','BOS'])
        list_.append([text[1],text[3]])

all_data = np.char.lower(list_)
train = all_data[0:int(0.8*len(list_))]
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]
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

M=len(words) + 1
K=len(uniq_class)
N=len(class_)
Ntest=len(Xtest)
Nvalid=len(Xvalid)


# In[3]:


def generate(trainmodify):
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


# In[4]:


Xttrans,train_labels=generate(train)
Xvtrans,valid_labels=generate(valid)
Xtesttrans,test_labels=generate(test)


# In[5]:


def Indi (a , b):
    if a == b :
        return 1
    else :
        return 0


# In[6]:


def SGD(Xttrans,train_labels,theta_init,max_epoch):
    theta=theta_init
    nllvalid=[]
    nlltrain=[]
    epoch = 0
    while epoch < max_epoch:
        nll=0
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


# In[7]:


theta,grad,nlltrain,nllvalid=SGD(Xttrans,train_labels,theta_init,max_epoch)


# In[8]:


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


# In[9]:


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


# In[10]:


print (errtrain/N,' and ',errtest/Ntest)


# In[12]:


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
