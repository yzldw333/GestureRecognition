import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *
from Get_VIVA_HandGesture_Database import *

batchsize = 1
model_path = './train_model/'

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
    if os.path.exists(model_path)==False:
        os.mkdir(model_path)
    graph = tf.Graph()
    with graph.as_default():
        wb=np.sqrt(6/(3*57*125*32+28*51*119*4))
        W_conv3d_1 = weight_variable_u([5,7,7,3,4],-wb,wb)
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
        b_fc1 = bias_variable([512])
        W_fc2 = weight_variable([512,256])
        b_fc2 = bias_variable([256])
        W_fc3 = weight_variable([256,19])
        b_fc3 = tf.Variable(tf.zeros([19]))
        tf.summary.histogram('W_conv3d_1',W_conv3d_1)
        tf.summary.histogram('W_fc_1', W_fc1)
        tf.summary.histogram('W_fc_2', W_fc2)
        with tf.name_scope('Inputs') as scope:
            X = tf.placeholder('float',shape=[batchsize,32,57,125,3])
            y_ = tf.placeholder('int64',shape=[batchsize])
            tf.summary.histogram('input',X)
        with tf.name_scope('Conv1') as scope:
            h_conv1 = conv3d_1(X,W_conv3d_1,b_conv3d_1)
            h_pool1 = max_pool_3d_1(h_conv1)
        with tf.name_scope('Conv2') as scope:
            h_conv2 = conv3d_2(h_pool1,W_conv3d_2,b_conv3d_2)
            h_pool2 = max_pool_3d_2(h_conv2)
        with tf.name_scope('Conv3') as scope:
            h_conv3 = conv3d_3(h_pool2,W_conv3d_3,b_conv3d_3)
            h_pool3 = max_pool_3d_3(h_conv3)
        with tf.name_scope('Conv4') as scope:
            h_conv4 = conv3d_4(h_pool3,W_conv3d_4,b_conv3d_4)
            h_pool4 = max_pool_3d_4(h_conv4)
            h_flat4 = tf.reshape(h_pool4,[-1,2*3*2*64])
        with tf.name_scope('Fc1') as scope:
            h_fc1 = fc(h_flat4,W_fc1,b_fc1)
        with tf.name_scope('Fc2') as scope:
            h_fc2 = fc(h_fc1,W_fc2,b_fc2)

        y_conv = tf.matmul(h_fc2,W_fc3)+b_fc3
        tf.summary.histogram('softmax_prob',tf.nn.softmax(y_conv))

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=y_, logits=y_conv, name='xentropy'),name='xentropy_mean')
        tf.summary.scalar('cross_entropy',cross_entropy)
        # # Create your variables
        # weights = tf.get_variable('weights', collections=['variables'])
        #
        # with tf.variable_scope('weights_norm') as scope:
        #     weights_norm = tf.reduce_sum(
        #         input_tensor=0.005 * tf.pack(
        #             [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
        #         ),
        #         name='weights_norm'
        #     )
        #
        # # Add the weight decay loss to another collection called losses
        # tf.add_to_collection('losses', weights_norm)
        #
        # # Add the other loss components to the collection losses
        # tf.add_to_collection('cross_entropy',cross_entropy)
        #
        # # To calculate your total loss
        # tf.add_n(tf.get_collection('losses'), name='total_loss')

        with tf.name_scope('train') as scope:
            train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
        with tf.name_scope('predict') as scope:
            correct_prediction = tf.equal(tf.argmax(y_conv,1),y_)
        with tf.name_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

    with tf.Session(graph=graph) as sess:

        seqs1,label1 = Read_TFRecord('train.tfrecords',None)
        #seqs2,label2 = Read_TFRecord('test.tfrecords',None)
        coord = tf.train.Coordinator()
        seqs_batch, label_batch1 = tf.train.shuffle_batch([seqs1,label1],batchsize,1060,1000,num_threads=4)
        mean, variance = tf.nn.moments(seqs_batch, [0,1,2,3])
        seqs_batch1 = (seqs_batch-tf.reshape(mean,[1,1,1,1,3]))/tf.sqrt(tf.reshape(variance,[1,1,1,1,3]))
        #seqs_batch2,label_batch2 = tf.train.shuffle_batch([seqs2,label2],batchsize,3000,1500,num_threads=5)
        merged_summary_op = tf.summary.merge_all()
        tf.local_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        summary_writer = tf.summary.FileWriter('./graph/logs', sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                for i in range(20000):
                    train_seqs, train_label= sess.run([seqs_batch1, label_batch1])
                    summary_str,_,_cross_entropy=sess.run((merged_summary_op,train_step,cross_entropy), feed_dict={X: train_seqs.reshape([batchsize, 32, 57, 125, 3]),
                                                    y_: train_label.reshape([batchsize])})
                    print(_cross_entropy)
                        # print("train accuracy")
                        # print(sess.run(accuracy,feed_dict={X: train_seqs.reshape([batchsize, 32, 57, 125, 2]),
                        #                             y_: train_label.reshape([batchsize])}))
                    #if i%100==0 and i!=0:

                    summary_writer.add_summary(summary_str,i)

                    if i%1000==0 and i!=0:
                        saver.save(sess=sess,save_path=model_path,global_step=i)
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


