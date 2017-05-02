import cv2
import numpy as np
import matplotlib.pyplot as plt

GRAD=1
NORMAL=2
GRAD_NORMAL=3

def Drop_Repeat_Frames(seqs, frameNum):
    '''
    Drop and Repeat Frames to make each video has same frame num.
    :param seqs: video sequences
                numpy array
    :param frameNum: frame num
                int
    :return: new video sequences
                numpy array
    '''
    length = np.size(seqs,0)
    shape = seqs.shape
    if len(shape) == 4 :
        seqs = seqs.reshape([length,shape[-3],shape[-2],shape[-1]])
        counts = shape[-3]*shape[-2]*shape[-1]
    elif len(shape) == 3:
        seqs = seqs.reshape([length,shape[-2],shape[-1],1])
        counts = shape[-2]*shape[-1]
    else:
        print("Shape ERROR!")
        return

    shape = seqs.shape
    newSeqs = []
    for i in range(frameNum):
        idx = (int)(i*1.0/frameNum*length+0.5)
        if idx>=length:
            idx=length-1
        newSeqs.extend(list(seqs[idx].ravel()))
    newSeqs = np.array(newSeqs,dtype=np.float32).ravel()
    length = len(newSeqs)//counts
    if len(shape) == 4:
        seqs = newSeqs.reshape([length,shape[-3],shape[-2],shape[-1]])
    return seqs

def GetVideoSeq(name,color,style,height=100,width=100):
    '''
        use opencv to read video sequences
    :param name: video name like 'xxx.avi'
    :param color: is gray?
    :param style: normal/gradient image
    :return: numpy array with the shape of
            (length,channel,height,width)
            or
            (2,length,channel,height,width)
    '''
    cap = cv2.VideoCapture(name)
    seqShape = None
    if cap.isOpened() == False:
        return None
    seqs = []
    length = 0
    while(True):

        res, frame = cap.read()
        if res:
            if color=='gray':
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if style & NORMAL==NORMAL:
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                if len(frame.shape) == 3:
                    frame = frame.transpose([2,0,1])
                seqs.extend(list(frame.ravel()))
            if style & GRAD == GRAD:
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                grad1 = cv2.Sobel(frame,cv2.CV_64F,1,0)
                if len(grad1.shape)==3:
                    grad1 = grad1.transpose([2,0,1])
                grad2 = cv2.Sobel(frame,cv2.CV_64F,0,1)
                if len(grad2.shape)==3:
                    grad2 = frame.transpose([2,0,1])
                grad = np.append(grad1.ravel(),grad2.ravel())
                seqs.extend(list(grad.ravel()))
                if len(frame.shape)==3:
                    frame = frame.transpose([2,0,1])

            seqShape = frame.shape
            length += 1
        else:
            break
    seqs = np.array(seqs,dtype = np.float32)
    #only gray image
    if len(seqShape) == 2:
        if style&GRAD_NORMAL==GRAD_NORMAL:
            seqs = seqs.reshape([length,3,seqShape[0],seqShape[1]])
        elif style&GRAD==GRAD:
            seqs = seqs.reshape([length,2,seqShape[0],seqShape[1]])
        elif style&NORMAL==NORMAL:
            seqs = seqs.reshape([length, 1, seqShape[0], seqShape[1]])

    if len(seqs.shape)==4:
        seqs = seqs.transpose([0,2,3,1])

    return seqs


if __name__ == '__main__':
    #test Drop_Repeat_Frames
    # seqs = np.zeros([32,1,33,33],dtype=np.float32)
    # newSeqs = Drop_Repeat_Frames(seqs,20)
    # print(newSeqs.shape)

    #test GetVideoSeq
    seqs = GetVideoSeq('./gray/01_01_01.avi',style=GRAD_NORMAL,color="gray",height=57,width=125)

    seqs = Drop_Repeat_Frames(seqs,32)
    print(seqs.shape)
    for e in seqs:
        plt.imshow(e[:,:,0].reshape([57,125]),cmap=plt.cm.gray)
        plt.show()
    print(seqs.shape)
