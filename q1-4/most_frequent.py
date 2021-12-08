import os
from data import *
from collections import defaultdict


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    per_word_count = {}
    tag_count = {}
    for sent in train_data:
        for tup in sent:
            if tup[0] not in per_word_count:
                per_word_count[tup[0]] = {}
            if tup[1] not in per_word_count[tup[0]]:
                per_word_count[tup[0]][tup[1]] = 0
            per_word_count[tup[0]][tup[1]] += 1
            if tup[1] not in tag_count:
                tag_count[tup[1]] = 0
            tag_count[tup[1]] += 1

    common_tag = max(tag_count, key=tag_count.get)

    def default_tag():
        return common_tag
    d = defaultdict(default_tag)
    for key, val in per_word_count.items():
        d[key] = max(val, key=val.get)
    return d
    ### END YOUR CODE


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        raise NotImplementedError
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


if __name__ == "__main__":
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)
