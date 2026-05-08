import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FireLLMMetrics:
    """Metrics helper for FireLLM classifier and RAG trainer.

    Provides lightweight calculation and reporting for common
    classification metrics used across the project.
    """

    def __init__(self, output_dir: str = "./output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: Dict = {}
        self.confusion_matrix = None

    def _extract_arrays(self, true_labels, predictions):
        # true_labels: list of int OR list of dicts like {"fire_spread": 3}
        # predictions: list of dicts like {"fire_spread": {"predicted_class": 3, "probabilities": {2:0.1,3:0.8,4:0.1}}}
        if len(true_labels) == 0:
            return np.array([]), np.array([]), []

        y_true = []
        for t in true_labels:
            if isinstance(t, dict):
                v = t.get('fire_spread', t.get('fire_spread_label', None))
            else:
                v = t
            y_true.append(int(v))

        y_pred = []
        probs_list = []
        for p in predictions:
            if isinstance(p, dict) and 'fire_spread' in p:
                fs = p['fire_spread']
                pred = fs.get('predicted_class') if isinstance(fs, dict) else fs
                prob = fs.get('probabilities', {}) if isinstance(fs, dict) else {}
            else:
                # fallback: prediction is an int
                pred = p
                prob = {}
            y_pred.append(int(pred))
            probs_list.append(prob)

        classes = sorted(list(set(list(y_true) + list(y_pred))))
        return np.array(y_true), np.array(y_pred), classes, probs_list

    def calculate_metrics(self, true_labels: List, predictions: List) -> Dict:
        """Compute a set of classification metrics and store in self.metrics.

        Returns the metrics dict.
        """
        y_true, y_pred, classes, probs_list = self._extract_arrays(true_labels, predictions)

        if y_true.size == 0:
            self.metrics = {}
            return self.metrics

        total = len(y_true)
        overall_acc = float((y_true == y_pred).mean())

        # Per-class precision/recall/f1
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for cls in classes:
            tp = int(((y_true == cls) & (y_pred == cls)).sum())
            fp = int(((y_true != cls) & (y_pred == cls)).sum())
            fn = int(((y_true == cls) & (y_pred != cls)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        precision_macro = float(np.mean(precision_per_class)) if precision_per_class else 0.0
        recall_macro = float(np.mean(recall_per_class)) if recall_per_class else 0.0
        f1_macro = float(np.mean(f1_per_class)) if f1_per_class else 0.0

        # MSE on integer labels
        mse = float(np.mean((y_true - y_pred) ** 2)) if total > 0 else 0.0

        # WMSE (inverse-frequency weighting)
        unique, counts = np.unique(y_true, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        weights = np.array([1.0 / class_counts.get(int(t), 1) for t in y_true], dtype=float)
        if weights.sum() > 0:
            weights = weights * (len(weights) / weights.sum())
        wmse = float(np.sum(weights * ((y_true - y_pred) ** 2)) / len(weights)) if len(weights) > 0 else 0.0

        # Brier score (multi-class): need probability vectors for each sample (use zeros if missing)
        class_order = classes if classes else [2, 3, 4]
        probs_mat = np.zeros((total, len(class_order)), dtype=float)
        cls_to_idx = {c: i for i, c in enumerate(class_order)}
        for i, pdict in enumerate(probs_list):
            if not isinstance(pdict, dict):
                continue
            for c, v in pdict.items():
                if int(c) in cls_to_idx:
                    probs_mat[i, cls_to_idx[int(c)]] = float(v)

        one_hot = np.zeros_like(probs_mat)
        for i, t in enumerate(y_true):
            if int(t) in cls_to_idx:
                one_hot[i, cls_to_idx[int(t)]] = 1.0

        if probs_mat.size > 0:
            brier_per_sample = np.sum((probs_mat - one_hot) ** 2, axis=1)
            brier_score = float(np.mean(brier_per_sample))
        else:
            brier_score = 0.0

        # RPS (ranked probability score) - cumulative squared diff over K-1 thresholds
        if probs_mat.size > 0 and probs_mat.shape[1] > 1:
            cum_probs = np.cumsum(probs_mat, axis=1)
            cum_obs = np.zeros_like(cum_probs)
            for j, k in enumerate(class_order):
                cum_obs[:, j] = (y_true <= k).astype(float)
            rps_terms = (cum_probs[:, :-1] - cum_obs[:, :-1]) ** 2
            rps = float(np.mean(np.sum(rps_terms, axis=1)))
        else:
            rps = 0.0

        # Confusion matrix
        idx_map = {c: i for i, c in enumerate(class_order)}
        cm = np.zeros((len(class_order), len(class_order)), dtype=int)
        for t, p in zip(y_true, y_pred):
            if int(t) in idx_map and int(p) in idx_map:
                cm[idx_map[int(t)], idx_map[int(p)]] += 1

        self.confusion_matrix = {
            'classes': class_order,
            'matrix': cm.tolist()
        }

        self.metrics = {
            'overall': {'accuracy': overall_acc, 'total': int(total)},
            'per_class': {
                'classes': class_order,
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class,
                'counts': [int(class_counts.get(c, 0)) for c in class_order]
            },
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'mse': mse,
            'wmse': wmse,
            'brier_score': brier_score,
            'rps': rps,
            'confusion_matrix': self.confusion_matrix,
            'timestamp': self.timestamp
        }

        return self.metrics

    def print_metrics(self):
        if not self.metrics:
            self.logger.info("No metrics to print")
            return
        overall = self.metrics.get('overall', {})
        self.logger.info(f"Metrics summary - samples: {overall.get('total', 0)}")
        self.logger.info(f"  Accuracy: {overall.get('accuracy', 0.0):.4f}")
        self.logger.info(f"  Precision (macro): {self.metrics.get('precision_macro', 0.0):.4f}")
        self.logger.info(f"  Recall (macro): {self.metrics.get('recall_macro', 0.0):.4f}")
        self.logger.info(f"  F1 (macro): {self.metrics.get('f1_macro', 0.0):.4f}")

    def save_metrics(self, path: Optional[str] = None):
        path = path or os.path.join(self.output_dir, f"metrics_{self.timestamp}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Metrics saved to: {path}")
        except Exception as e:
            self.logger.warning(f"Failed to save metrics: {e}")

    def generate_detailed_report(self, predictions: Optional[List] = None, true_labels: Optional[List] = None):
        # If predictions and true_labels provided, write a detailed CSV; otherwise skip
        try:
            import pandas as pd
        except Exception:
            self.logger.warning("pandas not available, skipping detailed report")
            return

        if predictions is None or true_labels is None:
            self.logger.debug("No predictions/true_labels provided for detailed report")
            return

        rows = []
        for i, (p, t) in enumerate(zip(predictions, true_labels)):
            true_val = t['fire_spread'] if isinstance(t, dict) else t
            pred_val = p['fire_spread']['predicted_class'] if isinstance(p, dict) and 'fire_spread' in p else p
            probs = p['fire_spread'].get('probabilities', {}) if isinstance(p, dict) and 'fire_spread' in p else {}
            row = {'sample_id': i + 1, 'true': int(true_val), 'pred': int(pred_val)}
            for k, v in probs.items():
                row[f"prob_{k}"] = float(v)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, f"detailed_predictions_{self.timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Detailed predictions saved to: {csv_path}")

    def plot_confusion_matrices(self):
        if not self.confusion_matrix:
            self.logger.debug("No confusion matrix available to plot")
            return

        classes = self.confusion_matrix['classes']
        cm = np.array(self.confusion_matrix['matrix'], dtype=int)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2. if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        out_path = os.path.join(self.output_dir, f"confusion_matrix_{self.timestamp}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Confusion matrix saved to: {out_path}")
