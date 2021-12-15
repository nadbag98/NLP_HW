from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict


def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    total_tokens = 0

    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts, e_word_tag_counts_prev, e_word_tag_counts_next, prefix_tag_counts, suffix_tag_counts = \
        [defaultdict(lambda: defaultdict(int)) for i in range(5)]

    for sent in train_sents:
        sent = [("", "*"), ("", "*")] + sent
        for i in range(2, len(sent)):
            word = sent[i][0]
            tag = sent[i][1]

            q_tri_counts[(sent[i - 2][1], sent[i - 1][1], tag)] += 1
            q_bi_counts[(sent[i - 1][1], tag)] += 1
            q_uni_counts[tag] += 1
            e_word_tag_counts[word][tag] += 1

            # prev/next word tag pairs
            if i != 2:
                e_word_tag_counts_prev[sent[i - 1][0]][tag] += 1
            if i != len(sent) - 1:
                e_word_tag_counts_next[sent[i + 1][0]][tag] += 1

            # suffix/prefix tag pairs
            max_sub_len = min(len(word), 4)
            for sub_len in range(0, max_sub_len):
                prefix_tag_counts[word[:sub_len + 1]][tag] += 1
                suffix_tag_counts[word[-sub_len - 1:]][tag] += 1
            total_tokens += 1
        q_tri_counts[(sent[-2][1], sent[-1][1], "STOP")] += 1
        q_bi_counts[(sent[-1][1], "STOP")] += 1
    q_bi_counts[("*", "*")] = len(train_sents)
    q_uni_counts["*"] = len(train_sents)
    q_uni_counts["STOP"] = len(train_sents)

    extra_decoding_arguments['total_tokens'] = total_tokens
    extra_decoding_arguments['q_tri_counts'] = q_tri_counts
    extra_decoding_arguments['q_bi_counts'] = q_bi_counts
    extra_decoding_arguments['q_uni_counts'] = q_uni_counts
    extra_decoding_arguments['e_word_tag_counts'] = e_word_tag_counts
    extra_decoding_arguments['e_word_tag_counts_prev'] = e_word_tag_counts_prev
    extra_decoding_arguments['e_word_tag_counts_next'] = e_word_tag_counts_next
    extra_decoding_arguments['prefix_tag_counts'] = prefix_tag_counts
    extra_decoding_arguments['suffix_tag_counts'] = suffix_tag_counts
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
    for i in range(len(sent)):
        sent_tups = list(zip(sent, predicted_tags))
        features = extract_features(sent_tups, i)
        feat_vectorized = vectorize_features(vec, features)
        probs = logreg.predict_proba(feat_vectorized)
        predicted_tags[i] = index_to_tag_dict[np.argmax(probs)]
    ### END YOUR CODE
    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
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
        viterbi_pred = memm_viterbi(sent, logreg, vec,
                                    index_to_tag_dict,
                                    extra_decoding_arguments)
        viterbi_pred_tag_seqs.append(viterbi_pred)
        total_words_count += len(sent)
        for i in range(len(sent)):
            if true_tags[i] == viterbi_pred[i]:
                correct_viterbi_preds += 1
            if true_tags[i] == greedy_pred[i]:
                correct_greedy_preds += 1
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
