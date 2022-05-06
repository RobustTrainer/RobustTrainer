# coding=utf-8
"""
Evaluation Metrics
"""
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, accuracy_score, roc_curve, auc


def report_measurements_of_classifier(truths, predictions):
    # metrics in SBR prediction
    # print(confusion_matrix(truths, predictions))
    tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()
    PD = tp / (tp + fn)  # recall
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)  # precision
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)

    # MCC
    MCC = matthews_corrcoef(truths, predictions)

    # our metrics
    report = classification_report(truths, predictions, target_names=['correct', 'error'], output_dict=True)
    accuracy = accuracy_score(truths, predictions)

    fpr, tpr, thresholds = roc_curve(truths, predictions, pos_label=1)
    auc_val = auc(fpr, tpr)

    rs = (report["correct"]["precision"], report["correct"]["recall"], report["correct"]["f1-score"],
          report["error"]["precision"], report["error"]["recall"], report["error"]["f1-score"],
          report["weighted avg"]["precision"], report["weighted avg"]["recall"], report["weighted avg"]["f1-score"],
          accuracy, MCC, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, auc_val
          )

    return rs