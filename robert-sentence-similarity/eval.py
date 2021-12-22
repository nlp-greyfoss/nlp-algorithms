from datasets import load_metric

if __name__ == '__main__':
    metric = load_metric("accuracy", "precision", "recall", "f1")
    print(metric)

