
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score


# In[2]:


file ='ner_dataset.csv'
doc = open(file,'r')


# In[64]:


st = doc.read(100)


# In[4]:


rel = st.splitlines()


# In[5]:


max_epoch = 50


# In[6]:


list_=[]
for i in rel[1:]:
    text = i.split(sep=',')
    if len(text) == 4:
        list_.append([text[1],text[3]])


# In[7]:


all_data = np.char.lower(list_)
train = all_data[0:int(0.8*len(list_))]
valid = all_data[int(0.8*len(list_)):int(0.9*len(list_))]
test = all_data[int(0.9*len(list_)):int(len(list_))]


# In[8]:


def Indi (a , b):
    if a == b :
        return 1
    else :
        return 0


# In[9]:


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


# In[10]:


Xttrans=[]
class_num=[]
for i,example in enumerate(X):
    Xttrans.append(words.tolist().index(example))
    class_num.append(uniq_class.tolist().index(class_[i]))

Xvtrans=[]
classv_num=[]
for i,example in enumerate(Xvalid):
    Xvtrans.append(words.tolist().index(example))
    classv_num.append(uniq_class.tolist().index(class_valid[i]))

Xtesttrans=[]
classtest_num=[]
for i,example in enumerate(Xtest):
    Xtesttrans.append(words.tolist().index(example))
    classtest_num.append(uniq_class.tolist().index(class_test[i]))

theta_init=np.matrix(np.zeros([M,K]))


# In[11]:


epoch=0
#Train theta using SGD and print NLL
def SGD(Xttrans,class_,theta_init,max_epoch):
    theta=theta_init
    nllvalid=[]
    nlltrain=[]
    Nvalid=len(Xvtrans)
    epoch = 0
    while epoch < max_epoch:
        nll=0
        grad = np.matrix(np.empty([M,K]))

        for i,example in enumerate(Xttrans): #Over N
            denom=0
            num=0
            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))
            for j,k in enumerate(range(K)): #Over K
                num=np.exp(theta[example,k]+theta[-1,k])

                I=Indi(class_[i],uniq_class[k])

                ans=-(I - num/denom)
                grad[example,k] = ans
                grad[-1,k] = ans
            theta[example] = theta[example] - 0.5 * grad[example]

            theta[-1] = theta[-1] - 0.5 * grad[-1]
            theta_init=theta


        # Keep track of negative log likelihood on training data
        nll=0
        for i,example in enumerate(Xttrans):
            denom=0
            num=0
            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))

            for j,k in enumerate(range(K)):
                num=np.exp(theta[example,k]+theta[-1,k])
                nll=nll-Indi(class_[i],uniq_class[k])/N*np.log(num/denom)
        nlltrain.append(nll)

        # Keep track of negative log likelihood on validation data
        nll=0
        for i,example in enumerate(Xvtrans):
            denom=0
            num=0
            for z in range(K):
                denom=denom+(np.exp(theta[example,z]+theta[-1,z]))

            for j,k in enumerate(range(K)):
                num=np.exp(theta[example,k]+theta[-1,k])
                nll=nll-Indi(class_valid[i],uniq_class[k])/Nvalid*np.log(num/denom)
        nllvalid.append(nll)
        epoch = epoch + 1
        print ('Completed epoch: {}'.format(epoch))
    return theta,grad,nlltrain,nllvalid


# In[12]:


theta,grad,nlltrain,nllvalid=SGD(Xttrans,class_,theta_init,max_epoch)


# In[13]:


predtest=[]
errtest = 0
for i,example in enumerate(Xtesttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        num.append(theta[example,k]+theta[-1,k])
    predtest.append(uniq_class[np.argmax(num)])
    if predtest[i] != class_test[i]:
        errtest= errtest + 1

predtrain=[]
errtrain = 0
for i,example in enumerate(Xttrans):
    num=[]
    for j,k in enumerate(range(K)): #Over K
        num.append(theta[example,k]+theta[-1,k])
    predtrain.append(uniq_class[np.argmax(num)])
    if predtrain[i] != class_[i]:
        errtrain= errtrain + 1


# In[14]:


error_train = np.round(errtrain/N,decimals=6)
error_test = np.round(errtest/Ntest,decimals=6)


# In[15]:


error_test


# In[16]:


errtest


# In[49]:


print(confusion_matrix(class_test,predtest))


# In[54]:


print(score(class_test,predtest)[2])


# In[63]:


precision, recall, fscore, class_count = score(class_test, predtest, labels=uniq_class)
print ('\n Classification metrics: \n')
for i in range(uniq_class.shape[0]):
    print (uniq_class[i])
    print ('\tprecision: ' + '{0:.5f}'.format(precision[i]))
    print ('\trecall: ' + '{0:.5f}'.format(recall[i]))
    print ('\tfscore: ' + '{0:.5f}'.format(fscore[i]))
    print ('\tTotal datapoints under this class: {}'.format(class_count[i]))

print ('\nF1 MACRO: {0:.5f}'.format(np.mean(fscore)))


# In[55]:
