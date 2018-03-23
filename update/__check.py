
# coding: utf-8

# In[1]:



# coding: utf-8

# dependencies
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import data_utils
#import matplotlib.pyplot as plt
'''

# In[2]:
def f():
    print('1')
'''

X, Y, en_word2idx, en_idx2word, en_vocab, zh_word2idx, zh_idx2word, zh_vocab = data_utils.read_dataset('en_n_zh.pkl')
print(en_vocab)


# In[3]:


# data processing
def replace_sentence_with_unk(sentence,en_vocab):
    for x in sentence:
        for y in range(len(x)):
            if x[y]  not in en_vocab:
                x[y]='<ukn>'
# data padding
def data_padding(x, y=None, length = 16):
    for i in range(len(x)):
        #x[i] = x[i] + (length - len(x[i])) * [en_word2idx['<pad>']]
        #y[i] = [zh_word2idx['<go>']] + y[i] + [zh_word2idx['<eos>']] + (length-len(y[i])) * [zh_word2idx['<pad>']]
        x[i] = x[i] + (length - len(x[i])) * [en_word2idx['<pad>']]
        if y is not None:
            y[i] = y[i] + (length - len(y[i])) * [zh_word2idx['<pad>']]
import random
def generate_useless_sentence(X,vocab):
    useless_data=[]
    for i in range(len(X)):
        sentence_length=len(X[random.randint(0,len(X)-1)])
        temp_sentence=[random.randint(2,len(vocab)) for x in range(sentence_length)]
        useless_data.append(temp_sentence)
    return useless_data
def mix_data(source_sentences,useless_sentences):
    over_all_data=[]
    label_of_mix_data=[]
    for x in range(len(source_sentences)+len(useless_sentences)):
        if(random.randint(0,1) is 0):
            if len(source_sentences) is 0:
                continue
            over_all_data.append(source_sentences[0])
            label_of_mix_data.append(1)
            del source_sentences[0]
        else:
            if len(useless_sentences) is 0:
                continue
            over_all_data.append(useless_sentences[0])
            label_of_mix_data.append(0)
            del useless_sentences[0]
    return over_all_data,label_of_mix_data
#data_padding(X, Y)
#print(Y)
def process_data(X,en_vocab,Y,zh_vocab):
    en_useless=generate_useless_sentence(X,en_vocab)
    zh_useless=generate_useless_sentence(Y,zh_vocab)
    en_all,en_label=mix_data(X,en_useless)
    zh_all,zh_label=mix_data(Y,zh_useless)
    data_padding(en_all)
    data_padding(zh_all)
    return en_all,en_label,zh_all,zh_label
en_all,en_label,zh_all,zh_label=process_data(X,en_vocab,Y,zh_vocab)
en_all_train,  en_all_test, en_label_train, en_label_test = train_test_split(en_all, en_label, test_size = 0.1)


# In[4]:


n_inputs = 16  # MNIST
n_hidden0 = 32
n_hidden1 = 64
n_hidden2 = 32
#n_hidden3 = 64
#n_hidden4 = 32
n_outputs = 2


# In[5]:


#reset_graph()

X_var = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y_var = tf.placeholder(tf.int64, shape=(None), name="y") 


# In[6]:


with tf.name_scope("dnn"):
    hidden0 = tf.layers.dense(X_var, n_hidden0, name="hidden0",
                              activation=tf.nn.relu)
    hidden1 = tf.layers.dense(hidden0, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    #hidden3 = tf.layers.dense(hidden2,n_hidden3,name="hidden3",
                            # activation=tf.nn.relu)
    #hidden4 = tf.layers.dense(hidden3,n_hidden4,name='hidden4',
                            # activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


# In[7]:


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_var, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    softmax=tf.nn.softmax(logits=logits)
    softmax_mean=tf.reduce_mean(tf.slice(softmax,[0,1],[-1,1]))


# In[8]:


learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[9]:


#X, Y, en_word2idx, en_idx2word, en_vocab, zh_word2idx, zh_idx2word, zh_vocab 


# In[10]:


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, Y_var, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# In[11]:


#print(X)
#print([zh_idx2word[id] for id in range(len(zh_vocab))])


# In[12]:


#useless_data=generate_useless_sentence(X,zh_word2idx,zh_idx2word,zh_vocab)


# In[13]:


#[zh_idx2word[id] for id in useless_data[0]]
#print(len(useless_data[0]))


# In[14]:


#[random.randint(0,1) for x in range(100)]


# In[15]:


n_epochs = 1000001
batch_size = 50
n_batches = int(np.ceil(len(en_all) / batch_size))
init = tf.global_variables_initializer()


# In[16]:


def next_batch(en_all,en_vocab,batch_size):
    current_position=0
    while 1:
        if current_position <= len(en_all):
            yield en_all[current_position:current_position+batch_size],en_label[current_position:                                                                                current_position+batch_size]
        else:
            current_position=0
            yield en_all[current_position:current_position+batch_size],en_label[current_position:                                                                              current_position+batch_size]
        current_position+=batch_size


# In[17]:


get=next_batch(en_all,en_vocab,batch_size)
print(len(en_all_train))
print(batch_size)
checkpoint_path = "./data/check_sentence.ckpt"
saver = tf.train.Saver()
sentences=[[ '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['銷售', '銷售', '矛盾', '矛盾', '矛盾', '矛盾', '五', '五', '五', '五', '五', '五', '五', '五', '五', '五', '五', '老實說'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及'], ['圖書館', '圖書館', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '救', '救', '救', '救'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['站在', '銷售', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '五', '五', '五', '五', '五', '五', '五', '五', '五', '五'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['事情', '事情', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '冊', '矛盾', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '行李車', '行李車', '行李車'], ['冊', '矛盾', '矛盾', '矛盾', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '轟', '轟', '轟', '轟', '轟'], ['站在',  '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車'], ['銷售', '銷售', '比的', '矛盾', '矛盾', '矛盾', '矛盾', '感到', '感到', '感到', '感到', '救', '救', '救', '救', '救', '救', '救'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '轟', '轟', '轟'], ['銷售', '銷售', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['銷售', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['銷售', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['聽錯', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '時代', '矛盾', '矛盾', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '轟', '轟', '轟', '轟', '轟', '轟', '轟'], ['冊', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['圖書館', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '行李車', '聽錯', '行李車', '行李車', '行李車', '行李車', '行李車', '聽錯', '聽錯', '聽錯'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['想像', '小姐', '小姐', '時代', '時代', '矛盾', '矛盾', '矛盾', '矛盾', '救', '救', '救', '救', '救', '救', '救', '救', '救'], ['銷售', '銷售', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在', '躺在'], ['想像', '想像', '聽錯', '聽錯', '聽錯', '聽錯', '犯', '犯', '犯', '觸', '觸', '觸', '觸', '觸', '觸'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['站在', '站在', '矛盾', '矛盾', '矛盾', '矛盾', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '下次', '下次'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '聽錯', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['圖書館', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車'], [ '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '機', '機', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '轟', '轟', '轟', '轟', '轟', '轟', '轟', '轟', '轟', '轟'], ['聽錯', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '五', '五', '五', '五', '五', '五', '五', '五', '轟', '轟'], ['銷售', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['圖書館', '圖書館', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['待在', '待在', '症', '症', '不憤', '不憤', '會長', '會長', '會長', '會長', '會長', '會長', '會長', '會長', '說', '說', '說', '說'], ['冊', '冊', '捆', '捆', '捆', '行李車', '行李車', '行李車', '行李車', '行李車', '行李車', '躺在', '行李車', '行李車', '行李車', '兩個', '兩個', '兩個'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['冊', '冊', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯', '聽錯'], ['站在', '聽錯', '矛盾', '矛盾', '矛盾', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及', '以及'], ['站在', '借光', '捆', '矛盾', '矛盾', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯', '犯'], ['站在', '站在', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸', '觸'], ['冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾'], ['站在', '冊', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血', '冷血'], ['站在', '冊', '時代', '時代', '以及', '以及', '以及', '以及', '以及', '以及', '老實說', '老實說', '老實說', '老實說', '老實說', '老實說', '老實說', '老實說'], ['銷售', '銷售', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾', '矛盾']]

def check_sentence(sentences,zh_vocab,zh_word2idx):
    sentence=[]
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        accuracy_val = sess.run(accuracy,feed_dict={X_var: en_all_test, Y_var: en_label_test})
        print('accuracy {}',accuracy_val)
        if  isinstance(sentences[0][0],int) is not True:
            data_padding(sentences)
            #replace_sentence_with_unk(sentences,zh_vocab)
            li=[]
            replace_sentence_with_unk(sentences,zh_vocab)
            for _sentence in sentences:
                l=[zh_word2idx[word] for word in _sentence]
                _sentence=l
                li.append(l)
            next_li=[]
            for _sentence in li:
                if len(_sentence) < 16:
                    _sentence=_sentence+(16-len(_sentence))*[1]
                    next_li.append(_sentence)
                else:
                    del _sentence[16:]
                    next_li.append(_sentence)
            print(len(next_li))
            return sess.run(softmax_mean,feed_dict={X_var:next_li })
        else:
            for _sentence in sentences:
                if len(_sentence) < 16:
                    _sentence=_sentence+(16-len(_sentence))*[1]
                else:
                    del _sentence[16:]
            return sess.run(softmax_mean,feed_dict={X_var:sentences })
            #print(sentence)
            #l=[[en_word2idx[word] for word in sentence[0]]]
            #print(sess.run(logits,feed_dict={X_var:l }))
            #print(sess.run(softmax,feed_dict={X_var:l }))