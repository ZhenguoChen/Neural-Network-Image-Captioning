'''
Extract features from images
'''
import cv2
import numpy as np
from os import listdir
from scipy.io import savemat
import json

def surf_extractor():
    # get image name list
    # image_names = [im for im in listdir('./data/Flicker8k_Dataset')]
    dataset = json.load(open('../data/flickr8k/dataset.json'))
    image_names = []
    for image in dataset['images']:
        image_names.append(image['filename'])

    print 'number of images', len(image_names)

    # create feature extractor
    # surf = cv2.xfeatures2d.SURF_create()
    surf = cv2.ORB_create(128)

    # create features data
    feats = {'feats': [],
             '__globals__': [],
             '__header__': b'MATLAB 5.0 MAT-file Platform: Ubuntu, Zhenguo Chen',
             '__version__': '1.0'}

    # extract features from images
    for im_name in image_names:
        print im_name
        img = cv2.imread('./data/Flicker8k_Dataset/'+im_name)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # find the keypoints and descriptors with SIFT
        kp, des = surf.detectAndCompute(img,None)
        # print len(des.flatten())
        # set descriptor to the same size
        # flat = des.flatten()[:8000]
        flat = des.flatten()
        flat.resize(4096)

        # feats['feats'] = np.append(feats['feats'], flat)
        feats['feats'].append(flat.tolist())

    savemat('./data/Flicker8k_KNN/knn_feats.mat', feats)

def cnn_extractor(use_gpu=False, ):
    '''
    extract features using ILSVRC-2014 model (VGG team) with 16 weight layers
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    :return:
    '''

