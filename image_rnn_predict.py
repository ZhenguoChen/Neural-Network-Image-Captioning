from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch

def captioning(images):
    '''
    do feature extraction for each images, and use LSTM to generate captions
    :param images: image file names
    :return: captions
    '''
    model = Image_LSTM()
    model.load_weights('model/keras/checkpoint_0.h5')
    model.set_dict()
    captions = []

    feats = feature_extraction_batch(images)
    for feat in feats:
        caption = model.predict(feat)
        captions.append(caption)

    print captions
    return captions

if __name__ == '__main__':
    img1 = 'static/img/download.jpg'
    img2 = 'static/img/dog1.jpg'
    img3 = 'static/img/dog2.jpg'

    imgs = [img1, img2, img3]

    captioning(imgs)