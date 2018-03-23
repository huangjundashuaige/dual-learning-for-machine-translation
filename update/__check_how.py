
# coding: utf-8

# In[1]:



# coding: utf-8

# dependencies
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import data_utils
import matplotlib.pyplot as plt
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
            yield en_all[current_position:current_position+batch_size],en_label[current_position:\                                                                                current_position+batch_size]
        else:
            current_position=0
            yield en_all[current_position:current_position+batch_size],en_label[current_position:\                                                                               current_position+batch_size]
        current_position+=batch_size


# In[17]:


get=next_batch(en_all,en_vocab,batch_size)
print(len(en_all_train))
print(batch_size)
checkpoint_path = "/tmp/check_sentence.ckpt"
saver = tf.train.Saver()

def check_sentence(sentences,en_vocab):
    sentence=[]
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        accuracy_val = sess.run(accuracy,feed_dict={X_var: en_all_test, Y_var: en_label_test})
        print('accuracy.{}',accuracy_val)
        if  isinstance(sentences[0][0],int) is not True:
            data_padding(sentences)
            l=[[en_word2idx[word] for _sentence in sentences for word in _sentence]]
            return sess.run(softmax_mean,feed_dict={X_var:l })[1]
        else:
            for _sentence in sentences:
                if len(_sentence) < 16:
                    _sentence=_sentence+(16-len(_sentence))*1
            return sess.run(softmax_mean,feed_dict={X_var:sentences })[1]
    return None
            #print(sentence)
            #l=[[en_word2idx[word] for word in sentence[0]]]
            #print(sess.run(logits,feed_dict={X_var:l }))
            #print(sess.run(softmax,feed_dict={X_var:l }))
# In[18]:
def f1():
    pass

'''
n_epochs = 1001
batch_size = 50
n_batches = int(np.ceil(len(en_all_train) / batch_size))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(len(en_all_train) // batch_size):
            X_batch, Y_batch = next(get)
            #X_batch=tf.convert_to_tensor(X_batch)
            #Y_batch=tf.convert_to_tensor(Y_batch)
            #print(X_batch)
            sess.run(training_op, feed_dict={X_var: X_batch, Y_var: Y_batch})
            #print(1)
        accuracy_val \
        = sess.run(accuracy, feed_dict={X_var: en_all_test, Y_var: en_label_test})
        if epoch % 10 is 0:
            print(accuracy_val)


    saver.save(sess, checkpoint_path)
'''


# In[19]:






    # In[20]:

'''
    op=tf.reduce_sum(tf.slice([[1,3],[1,3]],[0,1],[-1,1]))
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        print(sess.run(softmax_mean,feed_dict={X_var:en_all_test }))
'''