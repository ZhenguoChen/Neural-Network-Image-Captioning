from DataReader import Dataset_RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Reshape
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Model
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers.core import Permute
import keras
import numpy as np
from time import time

def preProBuildWordVocab(sentence_iterator, word_count_threshold=30):
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    # build map from word to index
    ixtoword = {}
    # ixtoword[0] = '.'
    # set ixtoword[1] to start tag so that we can set the first word input as start
    ixtoword[0] = '#start#'
    wordtoix = {}
    #wordtoix['.'] = 0
    wordtoix['#start#'] = 0

    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    word_counts['#start#'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

'''
what our model looks like:
                   [0,0,0,....1,0,0,0]
                           ^
                           |
                    -----------------
                    |               |
                    |   LSTM        |
                    |               |
                    |               |
                    |---------------|
                      ^    ^    ^  ^
                      |    |    |  |
                    image sent mask pos
'''


class Image_LSTM:
    '''
    A LSTM model, get image features as input, and get caption as output
    '''
    def __init__(self, DIM_INPUT=4096, DIM_EMBED=256, DIM_HIDDEN=256, MAX_LEN=82, BATCH_SIZE=128, N_WORDS=2943):
        '''
        initialize the lstm model with word embedding and image embedding
        :param DIM_INPUT: input dimension, 4096
        :param DIM_EMBED: embedding dimension, 256
        :param DIM_HIDDEN: hidden layer dimension, 256
        :param BATCH_SIZE: training batch size, 128
        :param N_WORDS: number of words
        '''
        self.batch_size = BATCH_SIZE
        self.max_len = MAX_LEN
        sent_input = Input(shape=(1,))
        img_input = Input(shape=(DIM_INPUT,))
        mask_input = Input(shape=(N_WORDS,))    # mark the previous word
        position_input = Input(shape=(MAX_LEN,)) # mark the position of current word

        # sentence embedding layer, get one word, output a vector
        sent_embed_layer = Embedding(output_dim=DIM_EMBED, input_dim=N_WORDS, input_length=1)(sent_input)
        sent_embed_layer = Reshape((DIM_EMBED,))(sent_embed_layer)
        sent_embed_layer = concatenate([sent_embed_layer, position_input])
        sent_embed_layer = Dense(DIM_EMBED)(sent_embed_layer)

        # img_embed_layer = Embedding(output_dim=DIM_HIDDEN, input_dim=DIM_INPUT, input_length=DIM_INPUT)(img_input)
        # concatenate image and embedded word as input for LSTM
        img_sent_merge_layer = concatenate([img_input, sent_embed_layer, mask_input])
        img_sent_merge_layer = Reshape((1, DIM_HIDDEN+DIM_INPUT+N_WORDS))(img_sent_merge_layer)

        lstm = LSTM(512)(img_sent_merge_layer)
        # add dropout and batch normalization layer to prevent overfitting
        lstm = Dropout(0.25)(lstm)
        lstm = Dense(N_WORDS)(lstm)
        lstm = BatchNormalization()(lstm)
        out = Activation('softmax')(lstm)

        self.model = Model(input=[img_input, sent_input, mask_input, position_input], output=out)
        self.model.compile(loss='categorical_crossentropy',
                           # optimizer=RMSprop(lr=0.0001, clipnorm=1.))
                           optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=00))
        print('finish build model')
        print(self.model.summary())


    def set_data(self, feats, captions):
        '''
        set the dataset for rnn model to use, and shuffle dataset
        :param feats: image features
        :param captions: image captions
        :return: 
        '''
        self.feats = feats
        self.captions = captions
        self.index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(self.index)

    def set_dict(self):
        '''
        set the word to index and index to word dictionary
        :return: 
        '''
        self.ixtoword = np.load('data/ixtoword.npy').tolist()
        self.n_words = len(self.ixtoword)

    def train(self, epochs):
        # record the training time
        epoch_time = []
        total_feats = len(self.feats)
        for epoch in range(epochs):
            begin = time()
            # train in batch
            for start, end in zip(range(0, len(self.index), self.batch_size), range(self.batch_size, len(self.index)+1, self.batch_size)):
                # preprocessing data
                current_caption_ind = []    # captions in wordtoix form
                next_caption_ind = []       # the "real classification"
                position = []               # record the position of current word
                current_mask_matrix = []    # mask
                current_feats = []          # current feature input
                current_batch = 0

                # get training batch
                current_feat = np.array(feats[self.index[start:end]])
                current_captions = captions[self.index[start:end]]

                # get current caption
                for (cap, feat) in zip(current_captions, current_feat):
                    # add start tag to caption
                    cap = '#start# '+cap
                    # get current caption indices
                    indices = [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix]
                    current_caption_ind.extend(indices[:-1])
                    current_mask = np.zeros((len(indices)-1, n_words))

                    # get the next word as a category problem
                    for i in range(1, len(indices)):
                        #print('i',i)
                        next_cap = np.zeros((n_words))
                        next_cap[indices[i]] = 1
                        # set current position of words in caption
                        cur_position = np.zeros((self.max_len))
                        cur_position[i-1] = 1

                        if i < len(indices)-1:
                            current_mask[i,:] = np.logical_or(current_mask[i,:], current_mask[i-1,:])
                            current_mask[i, indices[i-1]] = 1
                        # next word is use as ground truth
                        next_caption_ind.append(next_cap)
                        position.append(cur_position)
                        current_feats.append(feat)
                        current_batch += 1

                    current_mask_matrix.extend(current_mask)

                result = self.model.fit([np.array(current_feats), np.array(current_caption_ind),
                                         np.array(current_mask_matrix), np.array(position)],
                                         np.array(next_caption_ind),
                                         batch_size=current_batch, epochs=1)
                # print(result.history['loss'][-1])
                print('epoch {0}, process {1}/{2}'.format(epoch, start, total_feats))

            # save weights for each epoch
            epoch_file = 'model/keras/checkpoint_{0}.h5'.format(epoch)
            self.model.save_weights(epoch_file)
            end = time()
            epoch_time.append('epoch {0} use time: {1}'.format(epoch, end-begin))
            print(epoch_time)

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def predict(self, image, caption_len = 15):
        '''
        predict the caption of a image
        :param image: input image features
        :return: caption
        '''
        mask = np.zeros((caption_len, self.n_words))
        caption = []
        for i in range(1, caption_len):
            if i == 1:
                cur_word = 0
            else:
                # take the prediction as next word
                cur_word = next_word

            # set mask
            mask[i,:] = np.logical_or(mask[i,:], mask[i-1,:])
            mask[i, cur_word] = 1

            # set current word position
            pos = np.zeros(self.max_len)
            pos[i-1] = 1

            pred = self.model.predict([np.array(image), np.array([cur_word]), np.array([mask[i-1,:]]), np.array([pos])])[0]
            # get the best word
            next_word = pred.argmax()

            # decode the output to sentences
            caption.append(self.ixtoword[next_word])
            if self.ixtoword[next_word] == '.':
                break

        return ' '.join(caption)

if __name__ == '__main__':
    # set model variables
    dim_embed = 256
    dim_hidden = 256
    dim_in = 4096
    batch_size = 128
    # batch_size = 5
    momentum = 0.9
    n_epochs = 20

    # prepare data
    dataset = Dataset_RNN('data/flickr30k')
    feats, captions = dataset.get_data()
    feats = feats
    captions = captions

    wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)

    np.save('data/ixtoword', ixtoword)
        

    n_words = len(wordtoix)
    maxlen = np.max([x for x in map(lambda x: len(x.split(' ')), captions)])
    # init lstm model
    image_lstm = Image_LSTM(dim_in, dim_hidden, dim_embed, maxlen, batch_size, n_words)
    image_lstm.set_data(feats, captions)
    image_lstm.train(epochs=n_epochs)

