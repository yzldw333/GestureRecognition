import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    newSeqs = None
    length = np.size(seqs,0)
    shape = seqs.shape
    counts = 1
    if len(shape) == 4 :
        seqs = seqs.reshape([length,shape[-3],shape[-2],shape[-1]])
        counts = shape[-3]*shape[-2]*shape[-1]
    elif len(shape) == 3:
        seqs = seqs.reshape([length,1,shape[-2],shape[-1]])
        counts = shape[-2]*shape[-1]
    else:
        print("Shape ERROR!")
        return

    shape = seqs.shape
    if length > frameNum:
        # need cut frames
        cutNum = length - frameNum
        while cutNum != 0:
            per = length//cutNum
            if per==1:
                per = 2
                cutNum = length//per
            for i in range(cutNum):
                if i == 0:
                    newSeqs = []
                    newSeqs.extend(list(seqs[:per-1].ravel()))
                else:
                    newSeqs.extend(list(seqs[per*i:per*i+per-1].ravel()))
            newSeqs.extend(list(seqs[per*cutNum:].ravel()))
            newSeqs = np.array(newSeqs,dtype=np.float32).ravel()
            length = len(newSeqs)//counts
            if len(shape) == 4:
                seqs = newSeqs.reshape([length,shape[-3],shape[-2],shape[-1]])
            elif len(shape) == 3:
                seqs = newSeqs.reshape([length,shape[-2],shape[-1]])
            cutNum = length-frameNum

    elif length<frameNum:
        # need repeat frames
        addNum = frameNum - length
        while addNum != 0:
            per = length//addNum
            for i in range(addNum):
                if i == 0:
                    newSeqs = []
                    newSeqs.extend(list(seqs[:per].ravel()))
                    if per-1<length:
                        newSeqs.extend(list(seqs[per-1].ravel()))
                else:
                    newSeqs.extend(list(seqs[per*i:per*i+per].ravel()))
                    if per*i+per<length:
                        newSeqs.extend(list(seqs[per*i+per].ravel()))
            newSeqs.extend(list(seqs[per*addNum:].ravel()))
            newSeqs = np.array(newSeqs,dtype=np.float32).ravel()
            length = len(newSeqs)//counts
            if len(shape) == 4:
                seqs = newSeqs.reshape([length,shape[-3],shape[-2],shape[-1]])
            elif len(shape) == 3:
                seqs = newSeqs.reshape([length,shape[-2],shape[-1]])
            addNum = frameNum - length


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
            if style == 'normal':
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                #frame = frame.transpose([2,0,1])
                seqs.extend(list(frame.ravel()))
            elif style == 'gradient image':
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
    if len(seqShape) == 3:
        if style=='normal':
            seqs = seqs.reshape([length,seqShape[0],seqShape[1],seqShape[2]])
        else:
            seqs = seqs.reshape([length,seqShape[0]*2,seqShape[1],seqShape[2]])
    elif len(seqShape) == 2:
        if style=='normal':
            seqs = seqs.reshape([length,1,seqShape[0],seqShape[1]])
        else:
            seqs = seqs.reshape([length,2,seqShape[0],seqShape[1]])

    if len(seqs.shape)==4:
        seqs = seqs.transpose([0,2,3,1])

    return seqs


if __name__ == '__main__':
    #test Drop_Repeat_Frames
    # seqs = np.zeros([32,1,33,33],dtype=np.float32)
    # newSeqs = Drop_Repeat_Frames(seqs,20)
    # print(newSeqs.shape)

    #test GetVideoSeq
    seqs = GetVideoSeq('./data/01_01_01.avi',style='gradient image',color="gray",height=57,width=125)

    seqs = Drop_Repeat_Frames(seqs,32)
    print(seqs.shape)
    for e in seqs:
        plt.imshow(e[:,:,0].reshape([57,125]),cmap=plt.cm.gray)
        plt.show()
    print(seqs.shape)
