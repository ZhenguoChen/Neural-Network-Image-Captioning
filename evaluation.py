from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch
from utils import accuracy_rnn
import json
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def captioning_local(images):
    '''
    do feature extraction for each images, and use LSTM to generate captions
    :param images: image file names
    :return: captions
    '''
    model = Image_LSTM()
    model.load_weights('model/keras/checkpoint_19.h5')
    model.set_dict()
    len_paths = len(images)

    feats = feature_extraction_batch(images)

    print("Predicting Image Captions")
    print('Predicting Image %d/%d' % (1, len_paths))
    for i, path in enumerate(images):
        if ((i + 1) % 50 == 0):
            print('Predicting Image %d/%d' % (i + 1, len_paths))
        images[path][0] = model.predict(feats[i])

    return(images)

def get_bleu_helper(img_key):
    '''
    Helper function for use in "sorted" when sorting BLEU scores
    :param img_key: A image path dict key 
    :return: BLEU score
    '''
    value = imgs.get(img_key)
    return value[2]

def display_images(sortedPaths):
    '''
    Displays information and actual image from BLEU score sorted list
    :param sortedPaths: Sorted BLEU score image path list
    :return: None - only prints to console and displays images
    '''

    for path in sortedPaths:
        print("\n----- %s Predicted -----" % path)
        print(imgs[path][0])

        print("\n----- Actual %s -----" % path)
        print('\n'.join([x for x in imgs[path][1]]))

        print("\nBLEU Score: ", imgs[path][2])

        image = mpimg.imread(path)
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    '''
    img1 = 'static/img/image10.jpg'
    img2 = 'static/img/image1.jpg'
    img3 = 'static/img/image11.jpg'
    '''
    img1 = 'eval/comp1.jpg'
    img2 = 'eval/comp2.jpg'
    img3 = 'eval/comp3.jpg'
    img4 = 'eval/comp4.jpg'
    img5 = 'eval/comp5.jpg'

    imgs = [img1, img2, img3, img4, img5]

    imgs = {}
    NUM_OF_IMGS = 250

    print("Loading Image Paths")
    #Fetch JSON dataset of images and associated data
    with open('data/flickr8k/dataset.json') as json_data:
        jd = json.load(json_data)

    #Create a img path dictionary - {image_path : [prediction, actual "raw" sentences, BLEU score]}
    print("Creating Image Dictionary")
    print('Adding Image %d/%d' % (1, NUM_OF_IMGS))
    for i, img in enumerate(jd['images'][:NUM_OF_IMGS]):
        if ((i + 1) % 50 == 0):
            print('Adding Image %d/%d' % (i + 1, NUM_OF_IMGS))
        raw_sentences = []
        for sentence in img['sentences']:
            raw_sentences.append(sentence['raw'])
        imgs['data/Flicker8k_Dataset/%s' % (img['filename'])] = ['x', raw_sentences, 0.0]

    #Add caption to img path dictionary
    imgs = captioning_local(imgs)

    #Add BLEU scores to img path dictionary
    print("Determining Image BLEU Scores")
    print('Scoring Image %d/%d' % (1, NUM_OF_IMGS))
    avg_accuracy = 0.0
    for i, img in enumerate(imgs):
        if ((i + 1) % 50 == 0):
            print('Scoring Image %d/%d' % (i + 1, NUM_OF_IMGS))
        imgs[img][2] = accuracy_rnn(imgs[img][0], imgs[img][1])
        avg_accuracy += imgs[img][2]
    print("Average Accuracy Achieved: ", avg_accuracy/len(imgs))

    #Sort image prediction on BLEU score achieved
    print("Sorting BLEU Scores")
    bleuSortedPaths = sorted(imgs, key=get_bleu_helper)

    #Display top scoring images and all associated information
    print("\n----- TOP 10 SCORING IMAGES -----")
    display_images(bleuSortedPaths[-10:-1])

    print("\n----- BOTTOM 10 SCORING IMAGES -----")
    display_images(bleuSortedPaths[:10])
