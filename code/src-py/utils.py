from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred, average='binary'):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    return {
        'recall' : recall,
        'precision': precision,
        'f1': f1,
    }