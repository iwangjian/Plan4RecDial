#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np


LABEL_ACTION_PATH = "data/label_action.txt"
LABEL_TOPIC_PATH = "data/label_topic.txt"

def load_labels(fp, lower_case=True):
    labels = {}
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, item in enumerate(fr):
            k = item.strip().lower() if lower_case else item.strip()
            labels[k] = idx
    return labels


def calc_accuracy(hyps, refs):
    """ 
    Calculate accuracy.
    hyps: predicts, type: List
    refs: groundtruths, type: List
    """
    assert len(hyps) == len(refs)
    #acc = accuracy_score(y_true=refs, y_pred=hyps)
    acc_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp == ref:
            acc_list.append(1)
        else:
            acc_list.append(0)
    acc = np.mean(acc_list)
    return acc


def calc_bi_accuracy(hyps, refs):
    """ 
    Calculate bigram accuracy.
    hyps: predicts, type: List
    refs: groundtruths, type: List[List]
    """
    assert len(hyps) == len(refs)
    acc_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp in ref:
            acc_list.append(1)
        else:
            acc_list.append(0)
    acc = np.mean(acc_list)
    return acc


def load_data(fp, is_gold=False, lower_case=True):
    ids = []
    actions = []
    topics = []
    
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            if is_gold:
                ids.append(int(sample["id"]))
            if "cur_action" in sample.keys() and "cur_topic" in sample.keys():
                act = sample["cur_action"].lower() if lower_case else sample["cur_action"]
                topic = sample["cur_topic"].lower() if lower_case else sample["cur_topic"]
                actions.append(act)
                topics.append(topic)
            elif "plans" in sample.keys():
                if lower_case:
                    act_sep = "[a]"
                    top_sep = "[t]"
                    plans = sample["plans"].lower()
                else:
                    act_sep = "[A]"
                    top_sep = "[T]"
                    plans = sample["plans"]
                act = plans.split(act_sep)[1].split(top_sep)[0].strip()
                topic = plans.split(top_sep)[1].split(act_sep)[0].strip()
                actions.append(act)
                topics.append(topic)
            else:
                raise KeyError("No valid keys!")
    assert len(actions) == len(topics)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels.get(act, "寒暄") for act in actions]
    topic_ids = [topic_labels.get(top, "NULL") for top in topics]

    if is_gold:
        assert len(ids) == len(actions)
        bi_actions = []
        bi_topics = []
        prev_id = -1
        for idx, cur_id in enumerate(ids):
            if cur_id == prev_id:
                bi_acts = [action_ids[idx-1], action_ids[idx]]
                bi_tops = [topic_ids[idx-1], topic_ids[idx]]
            else:
                bi_acts = [action_ids[idx]]
                bi_tops = [topic_ids[idx]]
            bi_actions.append(bi_acts)
            bi_topics.append(bi_tops)
            prev_id = cur_id
        return (action_ids, topic_ids, bi_actions, bi_topics)
    else:
        return (action_ids, topic_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    pred_actions, pred_topics = load_data(args.eval_file)
    gold_actions, gold_topics, gold_bi_actions, gold_bi_topics = load_data(args.gold_file, is_gold=True)
    #print("gold_bi_actions: {}".format(gold_bi_actions))
    #print("gold_bi_topics: {}".format(gold_bi_topics))

    # calc accuracy
    action_acc = calc_accuracy(pred_actions, gold_actions)
    topic_acc = calc_accuracy(pred_topics, gold_topics)

    # calc bi-accuracy
    action_bi_acc = calc_bi_accuracy(pred_actions, gold_bi_actions)
    topic_bi_acc = calc_bi_accuracy(pred_topics, gold_bi_topics)

    output_str = "Action ACC: %.2f%%\n" % (action_acc * 100)
    output_str += "Topic ACC: %.2f%%\n" % (topic_acc * 100)
    output_str += "Action Bi-ACC: %.2f%%\n" % (action_bi_acc * 100)
    output_str += "Topic Bi-ACC: %.2f%%" % (topic_bi_acc * 100)

    print(output_str)
