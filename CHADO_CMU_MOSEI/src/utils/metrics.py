import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    report = classification_report(y_true, y_pred, digits=4, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, macro_f1, report, cm
