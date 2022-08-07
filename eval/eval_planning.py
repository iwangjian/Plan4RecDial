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
    labels["None"] = -1
    return labels


def calc_accuracy(hyps, refs):
    """ 
    Calculate accuracy.
    hyps: predicts, type: List
    refs: groundtruths, type: List
    """
    assert len(hyps) == len(refs)
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

def load_eval_data(fp, lower_case=True):
    actions = []
    topics = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            act = sample["action"]
            topic = sample["topic"]
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(actions) == len(topics)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels.get(act, -1) for act in actions]
    topic_ids = [topic_labels.get(top, -1) for top in topics]
    
    return (action_ids, topic_ids)


def load_gold_data(fp, lower_case=True):
    ids = []
    actions = []
    topics = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            ids.append(int(sample["id"]))
            act = sample["action_path"][0]       # current action
            topic = sample["topic_path"][0]      # current topic
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(ids) == len(actions) == len(topics)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels[act] for act in actions]
    topic_ids = [topic_labels[top] for top in topics]
        
    bi_action_ids = []
    bi_topic_ids = []
    prev_id = -1
    for idx, cur_id in enumerate(ids):
        if cur_id == prev_id:
            bi_acts = [action_ids[idx-1], action_ids[idx]]
            bi_tops = [topic_ids[idx-1], topic_ids[idx]]
        else:
            bi_acts = [action_ids[idx]]
            bi_tops = [topic_ids[idx]]
        bi_action_ids.append(bi_acts)
        bi_topic_ids.append(bi_tops)
        prev_id = cur_id
    
    return (action_ids, topic_ids, bi_action_ids, bi_topic_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    pred_actions, pred_topics = load_eval_data(args.eval_file)
    gold_actions, gold_topics, gold_bi_actions, gold_bi_topics = load_gold_data(args.gold_file)
    
    # calculate accuracy
    action_acc = calc_accuracy(pred_actions, gold_actions)
    topic_acc = calc_accuracy(pred_topics, gold_topics)

    # calculate bi-accuracy
    action_bi_acc = calc_bi_accuracy(pred_actions, gold_bi_actions)
    topic_bi_acc = calc_bi_accuracy(pred_topics, gold_bi_topics)

    output_str = "Action ACC: %.2f%%\n" % (action_acc * 100)
    output_str += "Action Bi-ACC: %.2f%%\n" % (action_bi_acc * 100)
    output_str += "Topic ACC: %.2f%%\n" % (topic_acc * 100)
    output_str += "Topic Bi-ACC: %.2f%%" % (topic_bi_acc * 100)

    print(output_str)
