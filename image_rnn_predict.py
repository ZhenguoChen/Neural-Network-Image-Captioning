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
    model.load_weights('model/keras/checkpoint_19.h5')
    model.set_dict()
    captions = []

    for feat in feats:
        caption = model.predict(feat)
        captions.append(caption)
        print caption
    return captions

if __name__ == '__main__':

    img1 = 'static/img/image10.jpg'
    img2 = 'static/img/image1.jpg'
    img3 = 'static/img/image14.jpg'
    '''
    img1 = 'eval/comp1.jpg'
    img2 = 'eval/comp2.jpg'
    img3 = 'eval/comp3.jpg'
    img4 = 'eval/comp4.jpg'
    img5 = 'eval/comp5.jpg'
    '''
    imgs = [img1, img2, img3]

    captions = captioning(imgs)
    print captions
