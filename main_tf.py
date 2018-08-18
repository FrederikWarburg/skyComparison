import tensorflow as tf
import pickle
import cv2
import os
import os.path as path
from utils import predict
from model import dilation_model_pretrained
from datasets import CONFIG
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    # Choose between 'cityscapes' and 'camvid'
    dataset = 'cityscapes'
    cityName = 'Bangkok'
    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Build pretrained model and save it as TF checkpoint
    with tf.Session() as sess:

        # Choose input shape according to dataset characteristics
        input_h, input_w, input_c = CONFIG[dataset]['input_shape']

        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name('softmax:0')
        model = tf.reshape(model, shape=(1,) + CONFIG[dataset]['output_shape'])


        # Initialize parameters
        color = ('b', 'g', 'r')

        baseImagePath = '../../../datasets/datasets/'+cityName+'/'
        bIPlen = len(baseImagePath)
        pathDirs = os.listdir(baseImagePath)

        file = open('../../../datasets/datasets/'+cityName+'Segmentation/pathAndCorr.txt', 'a+')
        documentsNotOfInterest = ['description.json','overview_map.html','overview.json','quality.txt','quality.csv',
                                  '.~lock.quality.csv#','usedPanoids.txt','cityInfo.txt', 'paths.txt','._cityInfo.txt', '._description.json']

        alreadySegmented = os.listdir('../../../datasets/datasets/'+cityName+'Segmentation/' +cityName)

        for pathFolder in pathDirs:
            print("looking at ", pathFolder)
            if pathFolder not in documentsNotOfInterest:

                if pathFolder not in alreadySegmented:
                    print("investigating ", pathFolder)
                    pathDir = path.join(baseImagePath,pathFolder)
                    seqSetDirs = os.listdir(pathDir)
                    t1 = time.time()
                    for seqSetFolder in seqSetDirs:

                        if seqSetFolder not in documentsNotOfInterest:

                            sequencePath = path.join(pathDir,seqSetFolder)
                            datePath = path.join(sequencePath,'dates.txt')

                            dateFile = open(datePath)

                            dates = []
                            for date in dateFile:
                                dates.append(date[:-1])

                            dateFile.close()

                            for date in dates:

                                imageDirs = os.listdir(path.join(sequencePath,date))
                                constant = 0
                                if "days.txt" in imageDirs:
                                    constant = 1

                                prevHistArray = []

                                for i in range(len(imageDirs)-constant):

                                    input_image_path = path.join(sequencePath,date) + '/' + str(i) + '.png'

                                    # Read and predict on a test image
                                    input_image = cv2.imread(input_image_path) #640,600,3

                                    input_image_large = cv2.resize(input_image,(2048,1024),interpolation=cv2.INTER_AREA)
                                    input_tensor = graph.get_tensor_by_name('input_placeholder:0')
                                    predicted_image = predict(input_image_large, input_tensor, model, dataset, sess)

                                    # Convert colorspace (palette is in RGB) and save prediction result
                                    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
                                    predicted_image = cv2.resize(predicted_image,(640,600), cv2.INTER_AREA)

                                    # sky cannot be in the lower part of the image. More efficient to use this when predicting,
                                    # but very easy to put in here...

                                    predicted_image[400:,:,:] = np.zeros((200,640,3),dtype=int)
                                    kernelSize = 70
                                    kernel = np.ones((kernelSize, kernelSize), np.uint8)
                                    mask = cv2.morphologyEx(cv2.cvtColor(predicted_image, cv2.COLOR_RGB2GRAY), cv2.MORPH_CLOSE, kernel)


                                    histArray = []
                                    for channel, col in enumerate(color):
                                        hist = cv2.calcHist([input_image], [channel], mask, [255], [1, 256])
                                        hist = np.array(hist, dtype=int).flatten()
                                        hist = hist / np.linalg.norm(hist,2) # we are only interested in the color dist. not the size, since trees and other stuff can come inbetween frames.
                                        histArray.append(hist)

                                    if prevHistArray == []:
                                        prevHistArray = histArray
                                        prev_input_path = input_image_path
                                        prevIm = input_image
                                        prevMask = predicted_image
                                    else:
                                        corrArray = []
                                        for hist,prevHist in zip(histArray,prevHistArray):

                                            #format
                                            hist = np.array([hist],dtype=np.float32).T
                                            prevHist = np.array([prevHist], dtype=np.float32).T

                                            corr = cv2.compareHist(hist,prevHist, cv2.HISTCMP_CORREL)
                                            corrArray.append(corr)

                                        #if np.min(corrArray) < 0.1:
                                        file.write(input_image_path[bIPlen:] + ' ; ' + prev_input_path[bIPlen:] + ' ; ' + str(round(corrArray[0],2)) +
                                                   ' ; ' + str(round(corrArray[1],2)) + ' ; ' + str(round(corrArray[2],2)) + '\n')

                                        pathImages = "../../../datasets/datasets/" + cityName + "Segmentation/" + cityName +"/"+pathFolder+"/"+seqSetFolder+"/"+date

                                        if not os.path.exists(pathImages):
                                            os.makedirs(pathImages)

                                        plt.figure(figsize=(10,20))
                                        plt.subplot(3,2,1)
                                        plt.imshow(cv2.cvtColor(prevIm,cv2.COLOR_BGR2RGB))
                                        plt.axis('off')
                                        plt.title(str(i-1)+'.png')
                                        plt.subplot(3,2,2)
                                        plt.imshow(cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB))
                                        plt.axis('off')
                                        plt.title(str(i)+'.png')
                                        plt.subplot(3,2,3)
                                        plt.imshow(prevMask)
                                        plt.axis('off')
                                        plt.subplot(3,2,4)
                                        plt.imshow(predicted_image)
                                        plt.axis('off')

                                        plt.subplot(3,2, 5)
                                        for channel, col in enumerate(color):
                                            plt.plot(prevHistArray[channel],color =col)
                                            plt.ylim([0,0.5])

                                        plt.subplot(3,2, 6)
                                        for channel, col in enumerate(color):
                                            plt.plot(histArray[channel],color =col)
                                            plt.ylim([0, 0.5])

                                        plt.title(str(np.round(corrArray,2)))
                                        plt.savefig(pathImages + "/plot" + str(i) + '.png')

                                        plt.close()

                                        # Update prev data with current data.
                                        prevHistArray = histArray
                                        prev_input_path = input_image_path
                                        prevIm = input_image
                                        prevMask = predicted_image
                    print("Time ", np.round((time.time()-t1)/60), len(seqSetDirs))


        file.close()






