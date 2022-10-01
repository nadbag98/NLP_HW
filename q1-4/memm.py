from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict
import re


def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE

    extra_decoding_arguments["q_dict"] = [[{} for i in range(6)] for i in range(6)]
    extra_decoding_arguments["q_dict_viterbi"] = {}

    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE

    features['next_word'] = next_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['prefix_1'] = curr_word[0]
    features['prefix_2'] = curr_word[:2]
    features['prefix_3'] = curr_word[:3]
    features['prefix_4'] = curr_word[:4]
    features['suffix_1'] = curr_word[-1]
    features['suffix_2'] = curr_word[-2:]
    features['suffix_3'] = curr_word[-3:]
    features['suffix_4'] = curr_word[-4:]
    features['prev_tag_pair'] = str(prevprev_tag) + "_" + str(prev_tag)

    ### END YOUR CODE
    return features


def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1],
                                 prevprev_token[1])


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector
        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    tag_to_ind_dict = {tag: i for i, tag in index_to_tag_dict.items()}
    sent_words = [tup[0] for tup in sent]
    sent_tups = [list(pair) for pair in (zip(sent_words, predicted_tags))]

    for i in range(len(sent)):
        features = extract_features(sent_tups, i)
        u_ind, v_ind = tag_to_ind_dict[features["prevprev_tag"]], tag_to_idx_dict[features["prev_tag"]]
        q_key = (features["prevprev_word"], features["prev_word"], features["next_word"])
        probs = extra_decoding_arguments["q_dict"][u_ind][v_ind].get(q_key)

        if probs is None:
            feat_vectorized = vectorize_features(vec, features)
            probs = logreg.predict_proba(feat_vectorized)[0]
            extra_decoding_arguments["q_dict"][u_ind][v_ind][q_key] = probs
        sent_tups[i][1] = index_to_tag_dict[np.argmax(probs)]

    predicted_tags = [tup[1] for tup in sent_tups]

    ### END YOUR CODE
    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    sent_words = [tup[0] for tup in sent]
    pi_prev = defaultdict(int)
    pi_prev[('*', '*')] = 1
    bp = {}
    tag_to_idx_dict = invert_dict(index_to_tag_dict)
    tags = list(index_to_tag_dict.values())
    tags.remove('*')

    for i in range(len(sent)):

        pi_curr = defaultdict(int)

        u_tags = tags
        t_tags = tags

        prev_word = sent_words[i-1]
        curr_word = sent_words[i]

        if i < 2:
            if i < 1:
                u_tags = ['*']
                prev_word = '<st>'
            t_tags = ['*']
            prevprev_word = '<st>'
        else:
            prevprev_word = sent_words[i - 2]

        if i == len(sent)-1:
            next_word = '</s>'
        else:
            next_word = sent_words[i + 1]

        for v in tags:
            for u in u_tags:

                q_key = (curr_word, next_word, prev_word, prevprev_word, u)
                probs = extra_decoding_arguments["q_dict_viterbi"].get(q_key)

                if probs is None:
                    feats_for_t = [extract_features_base(curr_word, next_word, prev_word, prevprev_word, u, t) for t in
                                   t_tags]
                    feat_for_t_vectorized = vec.transform(feats_for_t)
                    probs = logreg.predict_proba(feat_for_t_vectorized)
                    extra_decoding_arguments["q_dict_viterbi"][q_key] = probs

                tu_probs = np.array([pi_prev[(t, u)] for t in t_tags]) * probs[:, tag_to_idx_dict[v]]
                max_t = np.argmax(tu_probs)
                pi_curr[(u, v)] = tu_probs[max_t]
                bp[(i, u, v)] = t_tags[max_t]
        pi_prev = pi_curr

    if len(sent) == 1:
        probs_v = np.array([pi_curr[('*', v)] for v in tags])
        predicted_tags[-1] = tags[np.argmax(probs_v)]

    else:
        end_uv = [(u, v) for u in tags for v in tags]
        probs_uv = np.array([pi_curr[(u, v)]] for u, v in end_uv)
        best_tags = np.argmax(probs_uv)
        predicted_tags[-2], predicted_tags[-1] = end_uv[best_tags]

    for k in range(len(predicted_tags) - 3, -1, -1):
        predicted_tags[k] = bp[(k + 2, predicted_tags[k + 1], predicted_tags[k + 2])]

    ### END YOUR CODE
    return predicted_tags


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []

    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE

        greedy_pred = memm_greedy(sent, logreg, vec,
                                  index_to_tag_dict,
                                  extra_decoding_arguments)
        greedy_pred_tag_seqs.append(greedy_pred)
        viterbi_pred = memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        viterbi_pred_tag_seqs.append(viterbi_pred)

        total_words_count += len(sent)
        for i in range(len(sent)):
            if true_tags[i] == viterbi_pred[i]:
                correct_viterbi_preds += 1
            if true_tags[i] == greedy_pred[i]:
                correct_greedy_preds += 1
            else:
                print("true tag: " + str(true_tags[i]) + ",  greedy_pred: " + str(greedy_pred[i]))

        ### END YOUR CODE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)

    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")