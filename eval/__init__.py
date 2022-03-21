import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from statistics import harmonic_mean


harmonic_mean = harmonic_mean


def test_stats(test_true_labels, test_results, eval_mode='macro', return_stats_by_class=True):

    acc = accuracy_score(test_true_labels, test_results)
    p = precision_score(test_true_labels, test_results, average=eval_mode)
    r = recall_score(test_true_labels, test_results, average=eval_mode)
    f1 = 2 * p * r / (p + r)
    f1 = f1_score(test_true_labels, test_results, average=eval_mode)
    target = np.sort(np.unique(test_true_labels)).astype(int).tolist()
    report = classification_report(test_true_labels, test_results,
                                   target_names=target, output_dict=True)

    if return_stats_by_class:
        stats_by_class = []
        for label in target:
            item = report[label]
            stats_by_class.append([item['precision'], item['recall'], item['f1-score'], item['support']])
        return acc, p, r, f1, stats_by_class

    return acc, p, r, f1


def gen_stats(labels, true_labels, debug=False, report=False):
    top1_labels = np.argmax(labels, axis=1) + 1# 0-indexed class

    sk_metric_avg_mode = 'macro'
    p = precision_score(true_labels, top1_labels, average=sk_metric_avg_mode, zero_division=0)
    r = recall_score(true_labels, top1_labels, average=sk_metric_avg_mode, zero_division=0)
    f1 = f1_score(true_labels, top1_labels, average=sk_metric_avg_mode, zero_division=0)
    acc = accuracy_score(true_labels, top1_labels)

    if debug:
        print(top1_labels[:100])
        print(true_labels[:100])
    if report:
        report = classification_report(top1_labels, true_labels,
                                       target_names=np.unique(true_labels), output_dict=True)
        return top1_labels, (acc, p, r, f1), report
    return top1_labels, (acc, p, r, f1)


def topk_results(labels, true_labels, milestones=[1,2,5,10]):
    def topk_acc(k, default_max=5):
        topkk = np.argsort(labels)[:, -default_max:] + 1
        topk_correct = [0] * len(k)
        corr = []
        for i, val in enumerate(true_labels):
            for idx, kval in enumerate(k):
                if val in topkk[i, -kval:]:
                    corr.append(i)
                    topk_correct[idx] += 1
        return np.array(topk_correct) / float(len(true_labels))

    results = topk_acc(milestones, default_max=np.max(milestones))
    return results