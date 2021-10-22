#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction


def calc_f1(hyps, refs):
    """ Calculate char-level f1 score """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in zip(hyps, refs):
        golden_response = "".join(golden_response)
        response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total if pred_char_total > 0 else 0
    r = hit_char_total / golden_char_total if golden_char_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def calc_bleu(hyps, refs):
    """ Calculate bleu 1/2 """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def calc_distinct(seqs):
    """ Calculate intra/inter distinct 1/2 """
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return (intra_dist1, intra_dist2, inter_dist1, inter_dist2)


def calc_knowledge_f1(hyps, knowledge_refs, knowledge_alls):
    """" Calculate knowledge f1 score """
    golden_total = 0.0
    pred_total = 0.0
    hit_total = 0.0
    for response, golden_kd, all_kd in zip(hyps, knowledge_refs, knowledge_alls):
        response = "".join(response)
        golden_total += len(golden_kd)
        for kd in golden_kd:
            if is_knowledge_hit(response, kd):
                hit_total += 1
        for kd in all_kd:
            if is_knowledge_hit(response, kd):
                pred_total += 1
    p = hit_total / pred_total if pred_total > 0 else 0
    r = hit_total / golden_total if golden_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def is_knowledge_hit(utterance, kg_obj, threshold=0.55):
    flag = False
    if kg_obj in utterance:
        flag = True
    else:
        # knowledge recall
        common = Counter(utterance) & Counter(kg_obj)
        hit_char_total = sum(common.values())
        golden_char_total = len(kg_obj)
        recall = hit_char_total / golden_char_total if golden_char_total > 0 else 0
        if recall >= threshold:
            flag = True
    return flag

def label_knowledge(utterance, kg_list, lower_case=True):
    gold_knowledge = []
    all_objs = set()
    for triple in kg_list:
        assert len(triple) == 3
        all_objs.add(triple[0].lower() if lower_case else triple[0])
        all_objs.add(triple[2].lower() if lower_case else triple[2])
    for obj in all_objs:
        if is_knowledge_hit(utterance, obj):
            gold_knowledge.append(obj)
    all_objs = list(all_objs)
    return all_objs, gold_knowledge


def load_data(fp, is_gold=False, lower_case=True):
    samples = []
    all_knowledges = []
    gold_knowledges = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            response = sample["response"].lower() if lower_case else sample["response"]
            resp = [tok for tok in response]   # token-level list
            samples.append(resp)
            if is_gold:
                knowledge = sample["knowledge_graph"]
                all, gold = label_knowledge(response, knowledge, lower_case=lower_case)
                all_knowledges.append(all)
                gold_knowledges.append(gold)
    if is_gold:
        assert len(samples) == len(all_knowledges)
        assert len(samples) == len(gold_knowledges)
        return (samples, all_knowledges, gold_knowledges)
    else:
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    preds = load_data(args.eval_file)
    refs, all_knowledges, ref_knowlwedges = load_data(args.gold_file, is_gold=True)
    assert len(preds) == len(refs)
    ################
    #for pred, ref, ref_kd in zip(preds, refs, ref_knowlwedges):
    #    print("pred: ", "".join(pred))
    #    print("gold: ", "".join(ref))
    #    print("gold_kd: ", ref_kd)
    #    print("\n")

    # calc f1
    f1 = calc_f1(preds, refs)

    # calc bleu
    bleu1, bleu2 = calc_bleu(preds, refs)

    # calc distinct
    _, _, inter_dist1, inter_dist2 = calc_distinct(preds)

    # calc knowledge f1
    kd_f1 = calc_knowledge_f1(preds, ref_knowlwedges, all_knowledges)

    output_str = "F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f\n" % bleu1
    output_str += "BLEU2: %.3f\n" % bleu2
    output_str += "DISTINCT1: %.3f\n" % inter_dist1
    output_str += "DISTINCT2: %.3f\n" % inter_dist2
    output_str += "Knowledge F1: %.2f%%" % (kd_f1 * 100)

    print(output_str)
