'''
knn model for image description
'''
from DataReader import DataSet
from sklearn.neighbors import BallTree
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

banner1 = "*" *10
banner2 = "-" *10
banner3 = "&" *5
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
            print 'predict image ', i
            for j, id in enumerate(ids):
                print 'closed image:', j, 'image id:', id
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    vals =  sentence['raw']
        if len(vals) <=1:
            print vals
            
    def majority(self, ind):
        """given a list of indicies return the majority label"""
        for i, ids in enumerate(ind):
            print banner1, 'predict image ', i, banner1
            for j, id in enumerate(ids):
                print banner2, 'closed image:', j, 'image id:', id, banner2
                tokens = self.trains['descriptions'][id]['tokens']
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    print sentence['raw']

    def eval(self, examples, descriptions):
        '''
        test knn model and display image
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)
        #choice = self.majority(self, ind)
        #print("best sentence {}".format(descriptions[choice]))
        for i, ids in enumerate(ind):
            print('{0} predict image {1} filename: {2} {0}'.format(banner1, i, descriptions[i]['filename']))
            for j, id in enumerate(ids):
                print('{0} closed image: {1} image id: {2} {0}'.format(banner2, j, id))
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    print sentence['raw']
            sentences =  descriptions[i]["sentences"]
            for sentence in sentences:
                print("{0} actual description: \n {1}\n {0}".format(banner3, sentence['raw']))
            img = mpimg.imread('data/Flicker8k_Dataset/'+descriptions[i]['filename'])
            print 'data/Flicker8k_Dataset/'+descriptions[i]['filename']
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    data = DataSet('data/Flicker8k_KNN')
    trains = data.get_trains()
    knn = KNN(trains)
    valids = data.get_valids()
    knn.eval(valids['feats'][:10], valids['descriptions'][:10])
    # knn.predict(valids['feats'][:10])