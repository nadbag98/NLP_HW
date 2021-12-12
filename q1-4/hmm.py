import math
import os
import time
from data import *
from collections import defaultdict, Counter


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: defaultdict(int))
    ### YOUR CODE HERE
    for sent in sents:
        sent = [("", "*"), ("", "*")] + sent
        for i in range(2, len(sent)):
            q_tri_counts[(sent[i - 2][1], sent[i - 1][1], sent[i][1])] += 1
            q_bi_counts[(sent[i - 1][1], sent[i][1])] += 1
            q_uni_counts[sent[i][1]] += 1
            e_word_tag_counts[sent[i][0]][sent[i][1]] += 1
            total_tokens += 1
        q_tri_counts[(sent[-2][1], sent[-1][1], "STOP")] += 1
        q_bi_counts[(sent[-1][1], "STOP")] += 1
    q_bi_counts[("*", "*")] = len(sents)
    q_uni_counts["*"] = len(sents)
    q_uni_counts["STOP"] = len(sents)
    e_tag_counts = q_uni_counts.copy()
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    lambda3 = 1 - lambda1 - lambda2

    def log_q(y_i_2, y_i_1, y_i_0):
        tri_count = q_tri_counts[(y_i_2, y_i_1, y_i_0)]
        tri_denom = q_bi_counts[(y_i_2, y_i_1)]
        bi_count = q_bi_counts[(y_i_1, y_i_0)]
        bi_denom = q_uni_counts[y_i_1]
        uni_count = q_uni_counts[y_i_0]
        tri_factor = 0
        if tri_denom != 0:
            tri_factor = tri_count / tri_denom
        return math.log(lambda1 * tri_factor +
                        lambda2 * (bi_count / bi_denom) +
                        lambda3 * (uni_count / total_tokens))

    keys = list(q_uni_counts.keys())
    keys.remove("*")
    keys.remove("STOP")
    tags = defaultdict(lambda: keys)
    tags[0] = ["*"]
    tags[1] = ["*"]
    # TODO - preprocessing
    predicted_tags = ["*", "*"] + predicted_tags
    sent = ["", ""] + list(sent)
    # We work with log probability to avoid values too close to zero
    pi_prev = {("*", "*"): 0}
    bp = {}
    for i in range(2, len(sent)):
        pi_curr = {}
        for v in tags[i]:
            e_x_v = e_word_tag_counts[sent[i]][v]
            if e_x_v < 1:
                continue
            log_e_x_v = math.log(e_x_v) - math.log(e_tag_counts[v])
            for u in tags[i-1]:
                curr_max = float('-inf')
                curr_argmax = ""
                for w in tags[i-2]:
                    if (w, u) not in pi_prev:
                        continue
                    log_q_vwu = log_q(w, u, v)
                    log_pi_prev_wu = pi_prev[(w, u)]
                    sum_log_prob = log_q_vwu + log_pi_prev_wu
                    if sum_log_prob > curr_max:
                        curr_max = sum_log_prob
                        curr_argmax = w
                if curr_max != float("-inf"):
                    pi_curr[(u, v)] = curr_max + log_e_x_v
                    bp[(i, u, v)] = curr_argmax
        pi_prev = pi_curr

    end_predictions = defaultdict(lambda: float("-inf"))
    for u in tags[-2]:
        for v in tags[-1]:
            end_predictions[(u, v)] = log_q(u, v, "STOP")

    max_end_pred = float("-inf")
    for u in tags[-2]:
        for v in tags[-1]:
            if (u,v) not in pi_curr:
                continue
            uv_pred = pi_curr[(u,v)] + end_predictions[(u,v)]
            if uv_pred > max_end_pred:
                max_end_pred = uv_pred
                predicted_tags[-2], predicted_tags[-1] = u, v

  #  predicted_tags[-2], predicted_tags[-1] = max(pi_curr, key=pi_curr.get)
    predicted_tags = predicted_tags[2:]
    for k in range(len(predicted_tags)-3, -1, -1):
        predicted_tags[k] = bp[(k+4, predicted_tags[k+1], predicted_tags[k+2])]
    ### END YOUR CODE
    return tuple(predicted_tags)


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        lambda1 = 0.7
        lambda2 = 0.2

        pred_tag_seqs.append(hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
        e_word_tag_counts, e_tag_counts, lambda1, lambda2))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
