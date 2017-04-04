'''
knn model for image description
'''
from DataReader import DataSet
from sklearn.neighbors import BallTree
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KNN:
    '''
    knn model for generating description for images
    each word in vocabulary has a classifier
    '''
    def __init__(self, trains, k=3):
        '''
        initial a knn classifier
        '''
        self.trains = trains
        self.k = k
        self.knn = BallTree(trains['feats'])
        print 'done training'

    def predict(self, examples):
        '''
        get the k nearest neighbors of given examples
        :param examples: images needed to be described
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)
        for i, ids in enumerate(ind):
            print 'predict iamge ', i
            for j, id in enumerate(ids):
                print 'closed image:', j, 'image id:', id
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    print sentence['raw']

    def eval(self, examples, descriptions):
        '''
        test knn model and display image
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)
        for i, ids in enumerate(ind):
            print 'predict iamge ', i, descriptions[i]['filename']
            for j, id in enumerate(ids):
                print 'closed image:', j, 'image id:', id
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    print sentence['raw']
            img = mpimg.imread('data/Flicker8k_Dataset/'+descriptions[i]['filename'])
            print 'data/Flicker8k_Dataset/'+descriptions[i]['filename']
            plt.imshow(img)
            plt.show()

if __name__ == '__main__':
    data = DataSet('data/flickr8k')
    trains = data.get_trains()
    knn = KNN(trains)
    valids = data.get_valids()
    knn.eval(valids['feats'][:10], valids['descriptions'][:10])
    # knn.predict(valids['feats'][:10])
