from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch
from utils import accuracy_rnn
import json
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

    feats = feature_extraction_batch(images)

    for i, path in enumerate(images):
        caption = model.predict(feats[i])
        captions.append(caption)

    return(captions)

def captioning_local(images):
    '''
    do feature extraction for each images, and use LSTM to generate captions
    :param images: image file names
    :return: captions
    '''
    model = Image_LSTM()
    model.load_weights('model/keras/checkpoint_19.h5')
    model.set_dict()
    captions = []

    print("Extracting Image Features")
    feats = feature_extraction_batch(images)

    for i, path in enumerate(images):
        print("\n----- Predicting %s -----" % path)
        caption = model.predict(feats[i])
        images[path][0] = caption
        print("----- Actual %s -----" % path)
        print('\n'.join([x for x in images[path][1]]))
        captions.append(caption)

    return(images)

if __name__ == '__main__':

    imgs = {}

    #Fetch JSON dataset of images and associated data
    with open('data/flickr8k/dataset.json') as json_data:
        jd = json.load(json_data)

    #Create a img path dictionary - {image_path : [prediction, actual "raw" sentences, BLEU score]}
    for img in jd['images'][:100]:
        raw_sentences = []
        for sentence in img['sentences']:
            raw_sentences.append(sentence['raw'])
        imgs['data/Flicker8k_Dataset/%s' % (img['filename'])] = ['x', raw_sentences, 0.0]

    images = captioning_local(imgs)
    avg_accuracy = 0.0

    for img in images:
        images[img][2] = accuracy_rnn(images[img][0], images[img][1])
        avg_accuracy += images[img][2]
        print("%s = " % (img), images[img][2])

    print("Average Accuracy: ", avg_accuracy/len(imgs))
