from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch
from pynlpl.search import AbstractSearchState, DepthFirstSearch, BreadthFirstSearch, IterativeDeepening, HillClimbingSearch, BeamSearch


class ReorderSearchState(AbstractSearchState):
    def __init__(self, tokens, parent=None):
        self.tokens = tokens
        super(ReorderSearchState, self).__init__(parent)

    def expand(self):
        # Operator: Swap two consecutive pairs
        l = len(self.tokens)
        for i in range(0, l - 1):
            newtokens = self.tokens[:i]
            newtokens.append(self.tokens[i + 1])
            newtokens.append(self.tokens[i])
            if i + 2 < l:
                newtokens += self.tokens[i + 2:]
            yield ReorderSearchState(newtokens, self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return " ".join(self.tokens)


class InformedReorderSearchState(ReorderSearchState):
    def __init__(self, tokens, goal=None, parent=None):
        self.tokens = tokens
        self.goal = goal
        super(ReorderSearchState, self).__init__(parent)

    def score(self):
        """Compute distortion"""
        totaldistortion = 0
        for i, token in enumerate(self.goal.tokens):
            tokendistortion = 9999999
            for j, token2 in enumerate(self.tokens):
                if token == token2 and abs(i - j) < tokendistortion:
                    tokendistortion = abs(i - j)
            totaldistortion += tokendistortion
        return totaldistortion

    def expand(self):
        # Operator: Swap two consecutive pairs
        l = len(self.tokens)
        for i in range(0, l - 1):
            newtokens = self.tokens[:i]
            newtokens.append(self.tokens[i + 1])
            newtokens.append(self.tokens[i])
            if i + 2 < l:
                newtokens += self.tokens[i + 2:]
            yield InformedReorderSearchState(newtokens, self.goal, self)


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

    goalstate = InformedReorderSearchState("A dog runs through a field chasing a ball .".split(' '))

    informedinputstate = InformedReorderSearchState(' '.join(captions).split(' '),goalstate)
    search = BeamSearch(informedinputstate, beamsize=3, graph=True, minimize=True, debug=False)

    solution = search.searchbest()

    print('****diff***')
    print '*with beam search:',solution
    print '*caption(without):',captions

    return solution

if __name__ == '__main__':
    img1 = 'static/img/dog5.jpg'
    #img2 = 'static/img/dog1.jpg'
    #img3 = 'static/img/dog2.jpg'

    #imgs = [img1, img2, img3]
    imgs = [img1]


    captioning(imgs)