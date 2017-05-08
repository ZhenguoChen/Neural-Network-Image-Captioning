import json
import scipy.io
from collections import defaultdict
import numpy as np
import pandas as pd

class DataSet:
    '''
    Read data from file and provide required data
    data file format: .json and .mat file
    '''
    def __init__(self, dir_name):
        '''
        read data from files and build dateset
        :param dir_name: dataset directory
        '''
        # load the text information of training image
        self.dataset = json.load(open(dir_name+'/dataset.json', 'r'))

        # load image feature data
        if dir_name == 'data/Flicker8k_KNN':
            images_data = scipy.io.loadmat(dir_name+'/knn_feats.mat')
        else:
            images_data = scipy.io.loadmat(dir_name+'/vgg_feats_tensorflow.mat')
        self.images = images_data['feats']

        # split the images into train/validation/test sets
        self.split = defaultdict(list)
        for im in self.dataset['images']:
            self.split[im['split']].append(im)

        # build vocabulary
        word_counts = {}
        for im in self.dataset['images']:
            sentences = im['sentences']
            for sentence in sentences:
                for token in sentence['tokens']:
                    word_counts[token] = word_counts.get(token, 0) + 1

        self.ixtoword = []
        self.wordtoix = {}
        index = 0
        for token in word_counts.keys():
            self.wordtoix[token] = index
            self.ixtoword.append(token)
            index += 1

        self.vocab_size = len(self.wordtoix)

        # convert sentences to a vector and store in 'vec'
        for im in self.dataset['images']:
            sentences = im['sentences']
            im['vec'] = self.sents2vec(sentences)


    def getSize(self, split):
        '''
        get the number of images in split set
        :param split: train or validation or test
        :return: size
        '''
        return len(self.split[split])

    def sents2vec(self, sentences):
        '''
        convert sentences to vector
        combine all sentences for a image to a vector
        :param sentences: 'sentences' filed in json file
        :return:vector representing the caption of image
        '''

        vec = np.zeros(self.vocab_size)
        for sentence in sentences:
            for token in sentence['tokens']:
                if token in self.wordtoix:
                    vec[self.wordtoix[token]] += 1

        return vec

    def get_trains(self):
        '''
        get training examples from the whole dataset
        :return:
        '''
        trains = {}
        trains['feats'] = []
        trains['descriptions'] = []
        for train in self.split['train']:
            # get training examples
            trains['feats'].append(self.images[train['imgid']])
            trains['descriptions'].append(train)

        return trains

    def get_valids(self):
        '''
        get validation examples from the whole dataset
        :return:
        '''
        valids = {}
        valids['feats'] = []
        valids['descriptions'] = []
        for valid in self.split['val']:
            # get validation examples
            valids['feats'].append(self.images[valid['imgid']])
            valids['descriptions'].append(valid)

        return valids

class Dataset_RNN:
    '''
    read data from .npy, including caption and image features for flickr30k
    '''
    def __init__(self, data_path):
        self.captions = pd.read_table(data_path+'/results_20130124.token', sep='\t', header=None, names=['image', 'caption'])
        self.features = np.load(data_path+'/feats.npy')

    def get_data(self):
        return self.features, self.captions['caption'].values
