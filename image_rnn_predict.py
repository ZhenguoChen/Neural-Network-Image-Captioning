from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction

def captioning(images):
    '''
    do feature extraction for each images, and use LSTM to generate captions
    :param images: image file names
    :return: captions
    '''
    model = Image_LSTM()
    model.load_weights()
    captions = []
    for img in images:
        feat = feature_extraction(img)
        caption = model.predict(feat)
        captions.append(caption)

    print captions

if __name__ == '__main__':
    img1 = 'static/dog.jpg'
    img2 = 'static/dog1.jpg'
    img3 = 'static/dog2.jpg'

    imgs = [img1, img2, img3]

    captioning(imgs)