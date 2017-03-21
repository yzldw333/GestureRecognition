import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *
from Get_VIVA_HandGesture_Database import *

batchsize = 20

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01,name='weights')
    return tf.Variable(initial)

def weight_variable_u(shape,left,right):
    initial = tf.random_uniform(shape,minval=left,maxval=right,name='weights')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0,shape=shape,name='bias')
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
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Conv3d_1') as scope:
            wb=np.sqrt(6/(2*57*125*32+28*51*119*4))
            W_conv3d_1 = weight_variable_u([5,7,7,2,4],-wb,wb)
            b_conv3d_1 = bias_variable([1,28,51,119,4])
        with tf.name_scope('Conv3d_2') as scope:
            wb=np.sqrt(6/(4*25*59*14+12*21*55*8))
            W_conv3d_2 = weight_variable_u([3,5,5,4,8],-wb,wb)
            b_conv3d_2 = bias_variable([1,12,21,55,8])
        with tf.name_scope('Conv3d_3') as scope:
            wb=np.sqrt(6/(8*10*27*6+32*4*6*23))
            W_conv3d_3 = weight_variable_u([3,5,5,8,32],-wb,wb)
            b_conv3d_3 = bias_variable([1,4,6,23,32])
        with tf.name_scope('Conv3d_4') as scope:
            wb=np.sqrt(6/(32*6*11*4+64*4*7*2))
            W_conv3d_4 = weight_variable_u([3,3,5,32,64],-wb,wb)
            b_conv3d_4 = bias_variable([1,2,4,7,64])
        with tf.name_scope('FC1') as scope:
            W_fc1 = weight_variable([64*2*3*2,512])
            b_fc1 = bias_variable([512])
        with tf.name_scope('FC2') as scope:
            W_fc2 = weight_variable([512,256])
            b_fc2 = bias_variable([256])
        with tf.name_scope('FC3') as scope:
            W_fc3 = weight_variable([256,19])
            b_fc3 = bias_variable([19])

        with tf.name_scope('Inputs') as scope:
            X = tf.placeholder('float',shape=[batchsize,32,57,125,2])
            y_ = tf.placeholder('int64',shape=[batchsize])

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
        with tf.name_scope('train') as scope:
            train_step = tf.train.MomentumOptimizer(learning_rate=0.005,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
        with tf.name_scope('predict') as scope:
            correct_prediction = tf.equal(tf.argmax(y_conv,1),y_)
        with tf.name_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

    with tf.Session(graph=graph) as sess:

        seqs1,label1 = Read_TFRecord('train.tfrecords',None)
        #seqs2,label2 = Read_TFRecord('test.tfrecords',None)
        coord = tf.train.Coordinator()
        seqs_batch1, label_batch1 = tf.train.shuffle_batch([seqs1,label1],batchsize,1060,1000,num_threads=4)
        #seqs_batch2,label_batch2 = tf.train.shuffle_batch([seqs2,label2],batchsize,3000,1500,num_threads=5)
        tf.local_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        summary_writer = tf.summary.FileWriter('./graph/logs', sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                for i in range(20000):
                    train_seqs, train_label = sess.run([seqs_batch1, label_batch1])

                    sess.run(train_step, feed_dict={X: train_seqs.reshape([batchsize, 32, 57, 125, 2]),
                                                    y_: train_label.reshape([batchsize])})

                    if i%5==0:
                        print("train cross_entropy:")
                        print(sess.run(cross_entropy, feed_dict={X: train_seqs.reshape([batchsize, 32, 57, 125, 2]),
                                                    y_: train_label.reshape([batchsize])}))
                        print("train accuracy")
                        print(sess.run(accuracy,feed_dict={X: train_seqs.reshape([batchsize, 32, 57, 125, 2]),
                                                    y_: train_label.reshape([batchsize])}))
                    if i%1000==0 and i!=0:
                        saver.save(sess=sess,save_path='./train_model/model',global_step=i)
                    print('%d step done' % i)


                    #if i!=0 and i%20 == 0 :
                       #test_seqs,test_label = sess.run([seqs_batch2,label_batch2])
                       #print("test accuracy:")
                       #print(sess.run(accuracy,feed_dict={X:test_val.reshape([batchsize,32,57,125,2]),y_:test_y.reshape([batchsize])}))
        #except tf.errors.OutOfRangeError:
         #   print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # Wait for threads to finish.

        summary_writer.flush()
        summary_writer.close()
    #coord.join(threads=threads)


