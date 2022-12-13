import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    maxLength = 40

    print("READING DATA")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data")
        data = np.load("data.npy")
    else:
        data = np.zeros((25000,maxLength),dtype = 'float32')

        dir = os.path.dirname(__file__)
        #Extract
        if not os.path.exists(os.path.join(dir,"reviews/")):
            with tarfile.open("reviews.tar.gz","r") as tarball:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tarball, os.path.join(dir,"reviews/"))

        file_list = glob.glob(os.path.join(dir,
                                        'reviews/pos/*'))
        file_list.extend(glob.glob(os.path.join(dir,
                                        'reviews/neg/*')))
        print("Parsing %s files" % len(file_list))
        review_idx = 0
        for f in file_list:
            with open(f, "r",encoding="utf-8") as openf:
                s = openf.read()
                no_punct = ''.join(c for c in s if c not in string.punctuation)
                no_punct = no_punct.split()
                newarray = [0]*maxLength
                for idx in range(min(maxLength,len(no_punct))):
                    word = no_punct[idx].lower()
                    if word in glove_dict:
                        newarray[idx] = glove_dict[word]
                    else:
                        newarray[idx] = glove_dict["UNK"]
                data[review_idx] = (np.array(newarray))
                review_idx += 1

        np.save("data",data)
        data = np.load("data.npy")
        
    print("data sample",data[:1])

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    #print(len(lines))

    if os.path.exists(os.path.join(os.path.dirname(__file__), "embeddings.npy")):
        print("loading saved parsed data")
        embeddings = np.load("embeddings.npy")
        word_index_dict = np.load("word_index_dict.npy")
    else:
        data = open("glove.6B.50d.txt",'r',encoding="utf-8")
        #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r')

        lines = data.readlines()

        embeddings = np.zeros((len(lines)+1,50),dtype = 'float32')

        word_index_dict = dict()

        embeddings[0] = np.zeros((50),dtype = 'float32')
        word_index_dict["UNK"] = 0

        for idx in range(len(lines)):
            line = lines[idx].strip().split(" ")
            embeddings[idx+1] = (np.array([float(x) for x in line[1:]])) # vector
            word_index_dict[line[0]] = idx + 1           # line[0] is word

        #np.save("embeddings",embeddings)
        #np.save("word_index_dict",word_index_dict)

        del lines # save memory

    return embeddings,word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    #Parameters
    maxLength = 40
    numClasses = 2
    vectorLength = 50 #word embedded length
    lstmUnits = 10    #Avoid overfitting
    layer_num = 2
    window_size = 1
	
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(),name = "dropout_keep_prob")

    labels = tf.placeholder(tf.float32, [batch_size, numClasses],name = "labels")
    input_data = tf.placeholder(tf.int32, [batch_size, maxLength],name = "input_data")

    data = tf.Variable(tf.zeros([batch_size, maxLength, vectorLength]),dtype=tf.float32)
    #data = tf.Variable(tf.zeros([batch_size, maxLength - 2*window_size, vectorLength]),dtype=tf.float32)

    data = tf.cast(tf.nn.embedding_lookup(glove_embeddings_arr,input_data),tf.float32) #Batchsize*maxLength*vectorLength

    conv1=tf.layers.conv2d(inputs=data,filters=1,kernel_size=[1,3],activation=tf.nn.relu)
    #for word_pos in range(window_size,maxLength-window_size):
    #    data[word_pos-window_size] = tf.reduce_mean(tf.slice(raw_data,[1,word_pos-window_size],[-1,window_size*2+1,-1]),1)


    '''
    #########bi-direction LSTM ############
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, vectorLength])
    data = tf.split(data, maxLength)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=0.75)

    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=0.75)
    value, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, data, dtype=tf.float32)
    
    weight = tf.Variable(tf.truncated_normal([2*lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    prediction = (tf.matmul(value[-1], weight) + bias)

    ########

    '''

    ######## Basic LSTM
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)  #Define lstmUnit
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob) #Regulization, avoid overfitting
    
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))

   
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    ######
    
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name = "accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels),name = "loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob,optimizer, accuracy, loss
