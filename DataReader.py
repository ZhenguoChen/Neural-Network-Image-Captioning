import json
import scipy.io
from collections import defaultdict

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
        images_data = scipy.io.loadmat(dir_name+'/vgg_feats.mat')
        self.images = images_data['feats']

        # split the images into train/validation/test sets
        self.split = defaultdict(list)
        for im in self.dataset['images']:
            self.split[im['split']].append(im)

    def getSize(self, split):
        '''
        get the number of images in split set
        :param split: train or validation or test
        :return: size
        '''
        return len(self.split[split])