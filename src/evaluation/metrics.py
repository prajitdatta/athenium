"""
metrics.py
──────────
Athenium — Production Evaluation Framework

Accuracy alone is insufficient for high-stakes financial risk classification.
A model that predicts LOW for everything achieves 61% accuracy on a dataset
where 61% of contracts are LOW risk — and is completely useless.

This module implements a metric suite covering:
    1. Classification quality  — macro/weighted F1, per-class breakdown
    2. Confidence calibration  — Expected Calibration Error (ECE)
    3. Error analysis          — directional decomposition (over vs under)
    4. Business impact         — escalation rate, auto-process accuracy
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import classification_report, confusion_matrix


RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
N_CLASSES   = len(RISK_LABELS)


# ── Classification Metrics ────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true:  List[int],
    y_pred:  List[int],
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    Full classification metric suite.

    Args:
        y_true:  Ground truth class indices  [0..3]
        y_pred:  Predicted class indices     [0..3]
        y_proba: Softmax probabilities       (n_samples, 4)
                 Required for ECE and overconfidence rate.

    Returns:
        Dict of all metrics, structured for MLflow logging.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=RISK_LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy":           report["accuracy"],
        "macro_f1":           report["macro avg"]["f1-score"],
        "macro_precision":    report["macro avg"]["precision"],
        "macro_recall":       report["macro avg"]["recall"],
        "weighted_f1":        report["weighted avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall":    report["weighted avg"]["recall"],
    }

    for label in RISK_LABELS:
        p = label.lower()
        metrics[f"{p}_f1"]        = report[label]["f1-score"]
        metrics[f"{p}_precision"] = report[label]["precision"]
        metrics[f"{p}_recall"]    = report[label]["recall"]
        metrics[f"{p}_support"]   = int(report[label]["support"])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_labels"] = RISK_LABELS

    if y_proba is not None:
        metrics["ece"]                = expected_calibration_error(y_true, y_proba)
        metrics["overconfidence_rate"] = overconfidence_rate(y_true, y_pred, y_proba)

    return metrics


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true:  List[int],
    y_proba: np.ndarray,
    n_bins:  int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    A model that says "95% confidence" should be correct 95% of the time.
    ECE measures the gap between stated confidence and actual accuracy.

        ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|

    Interpretation:
        ECE < 0.02  — excellent calibration
        ECE < 0.05  — acceptable for production  (Athenium: 0.031 ✓)
        ECE > 0.10  — significantly miscalibrated, apply temperature scaling

    Why calibration matters for Athenium:
        A miscalibrated model says 90% confidence on a contract it gets wrong.
        The escalation policy does not trigger (threshold = 0.85).
        An analyst never reviews it. The error reaches production.

    Reference: Guo et al. (2017) — https://arxiv.org/abs/1706.04599
    """
    y_true = np.array(y_true)
    y_pred = np.argmax(y_proba, axis=1)
    conf   = np.max(y_proba, axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        if not in_bin.any():
            continue
        acc_bin  = (y_pred[in_bin] == y_true[in_bin]).mean()
        conf_bin = conf[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)


def overconfidence_rate(
    y_true:               List[int],
    y_pred:               List[int],
    y_proba:              np.ndarray,
    confidence_threshold: float = 0.90,
) -> float:
    """
    Fraction of high-confidence predictions that are wrong.

    High-confidence errors are the most dangerous in production:
    the model suppresses human review on exactly the cases where
    it is most confidently wrong.

    Target: < 0.03 (3% error rate among high-confidence predictions).
    """
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    max_conf = np.max(y_proba, axis=1)

    high_conf = max_conf >= confidence_threshold
    if not high_conf.any():
        return 0.0

    return float((y_pred[high_conf] != y_true[high_conf]).mean())


# ── Business Impact ───────────────────────────────────────────────────────────

def escalation_analysis(
    y_true:    List[int],
    y_proba:   np.ndarray,
    threshold: float = 0.85,
) -> Dict:
    """
    Model the production escalation policy.

    Contracts where model confidence < threshold are routed to
    the analyst queue. This function computes the operational
    impact of that threshold across the full dataset.

    Key metric: zero CRITICAL contracts should be auto-processed
    incorrectly. All CRITICAL misclassifications must be caught
    by the escalation gate.

    Args:
        threshold: Confidence below which the contract is escalated.
                   Default 0.85 → ~11.7% escalation in production.
    """
    y_true   = np.array(y_true)
    y_pred   = np.argmax(y_proba, axis=1)
    max_conf = np.max(y_proba, axis=1)

    auto_mask   = max_conf >= threshold
    review_mask = ~auto_mask

    auto_acc = (
        float((y_pred[auto_mask] == y_true[auto_mask]).mean())
        if auto_mask.any() else 0.0
    )

    critical_mask   = y_true == 3
    critical_missed = int((auto_mask & critical_mask & (y_pred != y_true)).sum())

    return {
        "threshold":                         float(threshold),
        "auto_rate":                         float(auto_mask.mean()),
        "review_rate":                       float(review_mask.mean()),
        "auto_accuracy":                     auto_acc,
        "n_auto":                            int(auto_mask.sum()),
        "n_review":                          int(review_mask.sum()),
        "n_total":                           len(y_true),
        "critical_contracts_missed_in_auto": critical_missed,
    }


def error_decomposition(
    y_true: List[int],
    y_pred: List[int],
) -> Dict:
    """
    Decompose errors by direction.

    For risk classification, errors are asymmetric:
        Under-classification (HIGH → LOW):  dangerous — missed risk
        Over-classification  (LOW → HIGH):  costly — unnecessary review

    Track both separately to inform threshold and curriculum decisions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors      = y_true != y_pred
    under_class = (y_pred < y_true).sum()
    over_class  = (y_pred > y_true).sum()

    return {
        "total_errors":               int(errors.sum()),
        "error_rate":                 float(errors.mean()),
        "under_classification_n":     int(under_class),
        "over_classification_n":      int(over_class),
        "under_classification_rate":  float(under_class / len(y_true)),
        "over_classification_rate":   float(over_class  / len(y_true)),
        "confusion_matrix":           confusion_matrix(
            y_true, y_pred, labels=list(range(N_CLASSES))
        ).tolist(),
        "confusion_labels":           RISK_LABELS,
    }
