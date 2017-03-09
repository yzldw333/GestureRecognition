import os
from preprocess import *
import tensorflow as tf
def Get_VIVA_HandGesture_Database(type):
    '''
    Get intensity data of VIVA HandGesture Database
    :param type: 'train' or 'test'
    :return: data=[(path,label),...]
    '''
    path = 'E:/tfProject/3DConvForGestureRecognition/data'
    choose_gestures = [1,2,3,4,6,7,8,13,14,15,16,21,23,27,28,29,30,31,32]
    out_data = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            if filename[-3:]!='avi':
                continue
            sample = int(filename[:2])//2
            gesture = int(filename[3:5])
            if gesture==80:
                gesture = 8
            if gesture==70:
                gesture = 7
            if type=='train' and sample != 0:
                for i in range(len(choose_gestures)):
                    if gesture == choose_gestures[i]:
                        e = (os.path.join(parent,filename),i)
                        out_data.append(e)
            if type=='test' and sample ==0:
                for i in range(len(choose_gestures)):
                    if gesture == choose_gestures[i]:
                        e = (os.path.join(parent,filename),i)
                        out_data.append(e)

    return out_data

def Write_TFRecord(pairList,tfRecordPath):
    '''
    :param pairList: pairList=[(path,label),...]
    :param tfRecordPath: data file output path
    :return:True or False
    '''
    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    index = 1
    for (path,label) in pairList:
        seqs = GetVideoSeq(name=path,color='gray',style='gradient image',height=57,width=125)
        if seqs is None:
            print('%s opened Error'%path)
            continue
        seqs = Drop_Repeat_Frames(seqs,32) #Fixed to 32 Frames
        seqs = np.array(seqs,dtype=np.uint8) #Reduce Space Usage
        raw_seqs = seqs.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'raw_seqs' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_seqs])),
                'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        writer.write(example.SerializeToString())
        print("%d done" % index)
        index+=1
    writer.close()

def Read_TFRecord(tfRecordPath):
    filename_queue = tf.train.string_input_producer([tfRecordPath])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
        'label':tf.FixedLenFeature([],tf.int64),
        'raw_seqs':tf.FixedLenFeature([],tf.string)})
    seqs = tf.decode_raw(features['raw_seqs'],tf.uint8)
    seqs = tf.reshape(seqs,[32,57,125,2])
    seqs = tf.cast(seqs,tf.float32)/255.0-0.5
    label = tf.cast(features['label'],tf.int64)
    return seqs,label

def Test_Read_TFRecord():
    seqs,label = Read_TFRecord('train.tfrecords')
    #coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess)
    sess.run(init)
    val,label = sess.run([seqs,label])
    print(val.shape,label)

    sess.close()

if __name__ == '__main__':
    pairList = Get_VIVA_HandGesture_Database('train')
    print(pairList)
    print(len(pairList))
    Write_TFRecord(pairList,'train.tfrecords')
    Test_Read_TFRecord()