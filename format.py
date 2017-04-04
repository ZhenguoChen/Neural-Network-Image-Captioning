'''
features of a image in vgg_feats.mat is stored as column.
in our program, features of a image is a row in the feature matrix
So, convert column into row
'''

import scipy.io

features = scipy.io.loadmat('data/flickr8k/vgg_feats.mat')

features_matrix = features['feats']
features_matrix2 = []

# convert column to row
for i in range(len(features_matrix[0])):
    features_matrix2.append(features_matrix[:,i])

features['feats']=features_matrix2

scipy.io.savemat('./data/flickr8k/vgg_feats_converted.mat', features)