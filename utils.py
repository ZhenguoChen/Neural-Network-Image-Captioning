'''
common functions used by both knn model and Neural Network
'''
from nltk.translate import bleu_score

def consensus(captions):
    '''
    get consensus caption
    :param captions: candidate captions
    :return: the consensus caption which maxmizes sum(sim(c,c')), for sim we use bleu score
    '''
    max = 0.0
    consensus = captions[0]

    # get the max bleu_score
    for caption in captions:
        references = captions[:]
        references.remove(caption)
        socre = bleu_score.sentence_bleu(references, caption)
        if caption > max:
            max = socre
            consensus = caption

    return consensus

def accuracy(predict, real):
    '''
    use bleu score as a measurement of accuracy
    :param predict: a list of predicted captions
    :param real: a list of actual descriptions
    :return: bleu accuracy
    '''
    accuracy = 0
    for i, pre in enumerate(predict):
        references = real[i]
        score = bleu_score.sentence_bleu(references, pre)
        accuracy += score

    return accuracy/len(predict)

def accuracy_rnn(predict, real):
    '''
    use bleu score as a measurement of accuracy
    :param predict: a list of predicted captions
    :param real: a list of actual descriptions
    :return: bleu accuracy
    '''
    accuracy = 0
    for r in real:
        score = bleu_score.sentence_bleu(r, predict)
        accuracy += score

    return accuracy/len(real)
