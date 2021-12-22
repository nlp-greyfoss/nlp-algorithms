import numpy as np
from datasets import load_metric


def compute_metrics(eval_preds):
    metric = load_metric("accuracy", "precision", "recall", "f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
