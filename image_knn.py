'''
knn model for image description
'''
from DataReader import DataSet
from utils import consensus
from utils import accuracy
from sklearn.neighbors import BallTree
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

banner1 = "*" *10
banner2 = "-" *10
banner3 = "&" *5

class KNN:
    '''
    knn model for generating description for images
    '''
    def __init__(self, trains, k=3):
        '''
        initial a knn classifier
        '''
        self.trains = trains
        self.k = k
        self.knn = BallTree(trains['feats'])
        print('done training')

    def predict(self, examples):
        '''
        get the k nearest neighbors of given examples
        :param examples: images needed to be described
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)

        consensus_captions = [] #consensus captions

        for i, ids in enumerate(ind):
            print('predict image ', i)
            captions = []  # candidate captions
            for j, id in enumerate(ids):
                # print 'closed image:', j, 'image id:', id
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    captions.append(sentence['raw'])

            consensus_captions.append(consensus(captions))

        return consensus_captions


    def eval(self, examples, descriptions):
        '''
        test knn model and display image
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)

        consensus_captions = []  # consensus captions
        actual_descriptions = []    # actual descriptions for all examples

        for i, ids in enumerate(ind):
            print('{0} predict image {1} filename: {2} {0}'.format(banner1, i, descriptions[i]['filename']))
            captions = []  # candidate captions
            for j, id in enumerate(ids):
                print('{0} closest image: {1} image id: {2} {0}'.format(banner2, j, self.trains['descriptions'][id]['filename']))
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    print sentence['raw']

                #get consensus captions
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    captions.append(sentence['raw'])


            print("Predictions: ", consensus(captions))

            consensus_captions.append(consensus(captions))

            sentences = descriptions[i]["sentences"]
            print('actual description:')
            actual_description = []
            for sentence in sentences:
                actual_description.append(sentence['raw'])
                print(sentence['raw'])
            actual_descriptions.append(actual_description)

            img = mpimg.imread('data/Flicker8k_Dataset/'+descriptions[i]['filename'])
            plt.imshow(img)
            plt.show()

        return consensus_captions, actual_descriptions

    def accuracy(self, examples, descriptions):
        '''
        using valid data to compute the accuracy of knn model
        :param examples:
        :param descriptions:
        :return:
        '''
        dst, ind = self.knn.query(examples, self.k)

        consensus_captions = []  # consensus captions
        actual_descriptions = []  # actual descriptions for all examples

        print('computing accuracy')

        for i, ids in enumerate(ind):
            captions = []  # candidate captions
            for j, id in enumerate(ids):

                # get consensus captions
                sentences = self.trains['descriptions'][id]['sentences']
                for sentence in sentences:
                    captions.append(sentence['raw'])

            consensus_captions.append(consensus(captions))

            sentences = descriptions[i]["sentences"]
            actual_description = []
            for sentence in sentences:
                actual_description.append(sentence['raw'])
            actual_descriptions.append(actual_description)

            if i%10 == 0:
                print("{0:.0f}%".format(float(i)/len(examples) * 100))

        print('100%')

        print(accuracy(consensus_captions, actual_descriptions))


if __name__ == '__main__':
    data = DataSet('data/flickr8k')
    trains = data.get_trains()
    knn = KNN(trains)
    valids = data.get_valids()
    predict, actual = knn.eval(valids['feats'][:10], valids['descriptions'][:10])
    predict = knn.predict(valids['feats'][:10])
    knn.accuracy(valids['feats'][:1], valids['descriptions'][:1])
    #acc = accuracy(predict, actual)
    print acc
