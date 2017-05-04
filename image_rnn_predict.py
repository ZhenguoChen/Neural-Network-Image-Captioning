from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch
import numpy as np

def captioning(images):
    '''
    do feature extraction for each images, and use LSTM to generate captions
    :param images: image file names
    :return: captions
    '''
    # extract feature for all images
    feats = feature_extraction_batch(images)
    # initialize LSTM model
    model = Image_LSTM()
    model.load_weights('model/keras/checkpoint_1.h5')
    model.set_dict()
    captions = []

    for feat in feats:
        caption = model.predict(feat)
        captions.append(caption)

    return captions

if __name__ == '__main__':
    img1 = 'static/img/image10.jpg'
    img2 = 'static/img/image1.jpg'
    img3 = 'static/img/image11.jpg'

    imgs = [img1, img2, img3]

    captioning(imgs)
