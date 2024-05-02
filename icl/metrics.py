from evaluate import load
import numpy as np
    
def ner_metrics(predictions, true_labels, labels_list):
    seqeval = load("seqeval")

    all_metrics = seqeval.compute(predictions=predictions, references=true_labels)

    metrics_values = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

    #keys = list(set([id2label[key].replace("B-", "").replace("I-", "") for key in list(id2label.keys()) if id2label[key] != "O"]))

    metrics_values_categories = {}

    for key in labels_list:
        metrics_values_categories[key] = {
            "precision": all_metrics[key]["precision"],
            "recall": all_metrics[key]["recall"],
            "f1": all_metrics[key]["f1"],
            "number": all_metrics[key]["number"],
        }

    return metrics_values, metrics_values_categories
