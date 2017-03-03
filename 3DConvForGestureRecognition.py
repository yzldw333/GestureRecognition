import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *
from Get_VIVA_HandGesture_Database import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def weight_variable_u(shape,left,right):
    initial = tf.random_uniform(shape,minval=left,maxval=right)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0,shape=shape)
    return tf.Variable(initial)

def conv3d_1(x,W,b):
    return tf.nn.relu(tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='VALID')+b)

def max_pool_3d_1(x):
    return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='VALID')

def conv3d_2(x,W,b):
    return tf.nn.relu(tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='VALID')+b)

def max_pool_3d_2(x):
    return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='VALID')

def conv3d_3(x,W,b):
    return tf.nn.relu(tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='VALID')+b)

def max_pool_3d_3(x):
    return tf.nn.max_pool3d(x,ksize=[1,1,1,2,1],strides=[1,1,1,2,1],padding='VALID')

def conv3d_4(x,W,b):
    return tf.nn.relu(tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='VALID')+b)

def max_pool_3d_4(x):
    return tf.nn.max_pool3d(x,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='VALID')

def fc(x,W,b):
    return tf.nn.relu(tf.matmul(x,W)+b)

if __name__ == '__main__':
    wb=np.sqrt(6/(2*57*125*32+28*51*119*4))
    W_conv3d_1 = weight_variable_u([5,7,7,2,4],-wb,wb)
    b_conv3d_1 = bias_variable([1,28,51,119,4])
    wb=np.sqrt(6/(4*25*59*14+12*21*55*8))
    W_conv3d_2 = weight_variable_u([3,5,5,4,8],-wb,wb)
    b_conv3d_2 = bias_variable([1,12,21,55,8])
    wb=np.sqrt(6/(8*10*27*6+32*4*6*23))
    W_conv3d_3 = weight_variable_u([3,5,5,8,32],-wb,wb)
    b_conv3d_3 = bias_variable([1,4,6,23,32])
    wb=np.sqrt(6/(32*6*11*4+64*4*7*2))
    W_conv3d_4 = weight_variable_u([3,3,5,32,64],-wb,wb)
    b_conv3d_4 = bias_variable([1,2,4,7,64])

    W_fc1 = weight_variable([64*2*3*2,512])
    W_fc2 = weight_variable([512,256])
    W_fc3 = weight_variable([256,19])
    b_fc1 = bias_variable([512])
    b_fc2 = bias_variable([256])
    b_fc3 = bias_variable([19])

    X = tf.placeholder('float',shape=[20,32,57,125,2])
    y_ = tf.placeholder('int64',shape=[20])

    h_conv1 = conv3d_1(X,W_conv3d_1,b_conv3d_1)
    h_pool1 = max_pool_3d_1(h_conv1)

    h_conv2 = conv3d_2(h_pool1,W_conv3d_2,b_conv3d_2)
    h_pool2 = max_pool_3d_2(h_conv2)

    h_conv3 = conv3d_3(h_pool2,W_conv3d_3,b_conv3d_3)
    h_pool3 = max_pool_3d_3(h_conv3)

    h_conv4 = conv3d_4(h_pool3,W_conv3d_4,b_conv3d_4)
    h_pool4 = max_pool_3d_4(h_conv4)
    h_flat4 = tf.reshape(h_pool4,[-1,2*3*2*64])

    h_fc1 = fc(h_flat4,W_fc1,b_fc1)
    h_fc2 = fc(h_fc1,W_fc2,b_fc2)

    y_conv = tf.nn.softmax(tf.matmul(h_fc2,W_fc3)+b_fc3)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_, logits=y_conv, name='xentropy'),name='xentropy_mean')
    train_step = tf.train.MomentumOptimizer(learning_rate=0.005,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    seqs,label = Read_TFRecord('train.tfrecords')
    test_seqs,test_label = Read_TFRecord('test.tfrecords')
    #coord = tf.train.Coordinator()
    val_batch, label_batch = tf.train.shuffle_batch([seqs,label],20,2000,1000,enqueue_many=False,num_threads=1)
    threads = tf.train.start_queue_runners(sess=sess)
    sess.run(init)
    for i in range(20000):
        val_batch,label_batch = sess.run([val_batch,label_batch])
        sess.run(train_step,feed_dict={X:val_batch.reshape([20,32,57,125,2]),y_:label_batch.reshape([20])})
        print('%d step done'%i)
        #if i!=0 and i%3 == 0 :
        #    test_val,test_label = sess.run([test_seqs,test_label])
        #    test_label = np.array([test_label],dtype=np.int64)
        #    print(sess.run(accuracy,feed_dict={X:test_val.reshape([1,32,57,125,2]),y_:test_label.reshape([1])}))
    sess.close()


