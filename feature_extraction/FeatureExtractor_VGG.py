'''
extract features using cnn
'''
import tensorflow as tf
import json
import scipy.io
from scipy.misc import imread, imresize
from CNN_VGG import vgg16

if __name__ == '__main__':
    # get image list
    dataset = json.load(open('../data/flickr8k/dataset.json'))
    image_names = []
    print 'opened dataset'
    for image in dataset['images']:
        image_names.append(image['filename'])

    # create features data
    feats = {'feats': [],
             '__globals__': [],
             '__header__': b'MATLAB 5.0 MAT-file Platform: Ubuntu, Zhenguo Chen',
             '__version__': '1.0'}

    # init vgg 16 layer cnn
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, '../model/vgg16_weights.npz', sess)

    # extract features from all image
    # extract features from images
    for im_name in image_names:
        print im_name
        # read raw image
        img = imread('../data/Flicker8k_Dataset/' + im_name, mode='RGB')
        img = imresize(img, (224, 224))
        # extract features
        feat = sess.run(vgg.probs, feed_dict={vgg.imgs: [img]})[0]
        feats['feats'].append(feat)

    scipy.io.savemat('../data/flickr8k/vgg_feats_tensorflow.mat', feats)