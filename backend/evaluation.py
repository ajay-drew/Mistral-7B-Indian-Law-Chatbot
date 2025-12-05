"""Evaluation metrics for model performance assessment.

This module provides comprehensive evaluation metrics including:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU scores
- Perplexity
- Accuracy
- F1 Score
- BERTScore
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    NLTK_BLEU_AVAILABLE = True
except ImportError:
    NLTK_BLEU_AVAILABLE = False
    logger.warning("NLTK BLEU not available. BLEU calculation will be limited.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available. ROUGE calculation will not work.")

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some metrics will not work.")

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")

try:
    import nltk
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some metrics may not work correctly.")


class EvaluationMetrics:
    """Comprehensive evaluation metrics for text generation models."""

    def __init__(self):
        """Initialize evaluation metrics."""
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        if NLTK_BLEU_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
        else:
            self.smoothing = None

    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

        Args:
            predictions: List of generated texts
            references: List of reference texts

        Returns:
            Dictionary with ROUGE scores for each metric
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            raise ImportError("rouge-score not available. Install with: pip install rouge-score")
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": {
                "precision": np.mean([self.rouge_scorer.score(r, p)["rouge1"].precision
                                     for p, r in zip(predictions, references)]),
                "recall": np.mean([self.rouge_scorer.score(r, p)["rouge1"].recall
                                  for p, r in zip(predictions, references)]),
                "f1": np.mean(rouge1_scores)
            },
            "rouge2": {
                "precision": np.mean([self.rouge_scorer.score(r, p)["rouge2"].precision
                                     for p, r in zip(predictions, references)]),
                "recall": np.mean([self.rouge_scorer.score(r, p)["rouge2"].recall
                                  for p, r in zip(predictions, references)]),
                "f1": np.mean(rouge2_scores)
            },
            "rougeL": {
                "precision": np.mean([self.rouge_scorer.score(r, p)["rougeL"].precision
                                     for p, r in zip(predictions, references)]),
                "recall": np.mean([self.rouge_scorer.score(r, p)["rougeL"].recall
                                  for p, r in zip(predictions, references)]),
                "f1": np.mean(rougeL_scores)
            }
        }

    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU scores.

        Args:
            predictions: List of generated texts
            references: List of reference texts (can be list of lists for multiple references)

        Returns:
            Dictionary with BLEU scores
        """
        if not NLTK_BLEU_AVAILABLE or self.smoothing is None:
            raise ImportError("NLTK not available. Install with: pip install nltk")
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref_tokens = ref.split()
            else:
                ref_tokens = ref
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothing)
            bleu_scores.append(score)

        return {
            "bleu": float(np.mean(bleu_scores)),
            "bleu_std": float(np.std(bleu_scores)),
            "bleu_min": float(np.min(bleu_scores)),
            "bleu_max": float(np.max(bleu_scores))
        }

    def calculate_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Calculate perplexity for given texts.

        Args:
            model: The language model
            tokenizer: The tokenizer
            texts: List of texts to evaluate
            device: Device to run on (defaults to model's device)

        Returns:
            Dictionary with perplexity metrics
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss)

        return {
            "perplexity": float(perplexity),
            "loss": float(avg_loss),
            "total_tokens": int(total_tokens)
        }

    def calculate_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        token_level: bool = False
    ) -> Dict[str, float]:
        """Calculate accuracy score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            token_level: If True, calculate token-level accuracy

        Returns:
            Dictionary with accuracy metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if token_level:
            # Token-level accuracy
            correct_tokens = 0
            total_tokens = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                min_len = min(len(pred_tokens), len(ref_tokens))
                correct_tokens += sum(1 for i in range(min_len) if pred_tokens[i] == ref_tokens[i])
                total_tokens += len(ref_tokens)
            accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        else:
            # Exact match accuracy
            exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
            accuracy = exact_matches / len(predictions) if len(predictions) > 0 else 0.0

        if token_level:
            return {
                "accuracy": float(accuracy),
                "correct_tokens": int(correct_tokens),
                "total_tokens": int(total_tokens)
            }
        else:
            return {
                "accuracy": float(accuracy),
                "exact_matches": int(exact_matches),
                "total": len(predictions)
            }

    def calculate_f1(
        self,
        predictions: List[str],
        references: List[str],
        token_level: bool = True
    ) -> Dict[str, float]:
        """Calculate F1 score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            token_level: If True, calculate token-level F1

        Returns:
            Dictionary with F1 metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if token_level:
            # Token-level F1
            all_pred_tokens = []
            all_ref_tokens = []
            for pred, ref in zip(predictions, references):
                all_pred_tokens.extend(pred.split())
                all_ref_tokens.extend(ref.split())

            # Create binary labels (1 if token matches, 0 otherwise)
            # This is a simplified version - in practice, you might want more sophisticated matching
            pred_set = set(all_pred_tokens)
            ref_set = set(all_ref_tokens)
            true_positives = len(pred_set & ref_set)
            false_positives = len(pred_set - ref_set)
            false_negatives = len(ref_set - pred_set)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            }
        else:
            # Use exact match as binary classification
            y_true = [1 if p.strip() == r.strip() else 0 for p, r in zip(predictions, references)]
            y_pred = [1] * len(y_true)  # All predictions are positive (exact match)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            return {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            }

    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en",
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Calculate BERTScore.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            lang: Language code (default: "en")
            device: Device to run on

        Returns:
            Dictionary with BERTScore metrics
        """
        if not BERTSCORE_AVAILABLE:
            raise ImportError("BERTScore not available. Install with: pip install bert-score")

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        P, R, F1 = bert_score(
            predictions,
            references,
            lang=lang,
            device=device,
            verbose=False
        )

        return {
            "bertscore_precision": float(P.mean().item()),
            "bertscore_recall": float(R.mean().item()),
            "bertscore_f1": float(F1.mean().item()),
            "bertscore_precision_std": float(P.std().item()),
            "bertscore_recall_std": float(R.std().item()),
            "bertscore_f1_std": float(F1.std().item())
        }

    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        model=None,
        tokenizer=None,
        include_bertscore: bool = True,
        include_perplexity: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict[str, any]:
        """Calculate all available metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            model: Optional model for perplexity calculation
            tokenizer: Optional tokenizer for perplexity calculation
            include_bertscore: Whether to include BERTScore (requires bert-score)
            include_perplexity: Whether to include perplexity (requires model and tokenizer)
            device: Device to run on

        Returns:
            Dictionary with all calculated metrics
        """
        results = {}

        # ROUGE scores
        try:
            results["rouge"] = self.calculate_rouge(predictions, references)
        except Exception as e:
            logger.warning(f"Failed to calculate ROUGE: {e}")
            results["rouge"] = None

        # BLEU scores
        try:
            results["bleu"] = self.calculate_bleu(predictions, references)
        except Exception as e:
            logger.warning(f"Failed to calculate BLEU: {e}")
            results["bleu"] = None

        # Accuracy
        try:
            results["accuracy"] = self.calculate_accuracy(predictions, references)
            results["accuracy_token"] = self.calculate_accuracy(predictions, references, token_level=True)
        except Exception as e:
            logger.warning(f"Failed to calculate accuracy: {e}")
            results["accuracy"] = None

        # F1 Score
        try:
            results["f1"] = self.calculate_f1(predictions, references)
        except Exception as e:
            logger.warning(f"Failed to calculate F1: {e}")
            results["f1"] = None

        # BERTScore (optional)
        if include_bertscore and BERTSCORE_AVAILABLE:
            try:
                results["bertscore"] = self.calculate_bertscore(predictions, references, device=device)
            except Exception as e:
                logger.warning(f"Failed to calculate BERTScore: {e}")
                results["bertscore"] = None
        else:
            results["bertscore"] = None

        # Perplexity (optional, requires model)
        if include_perplexity and model is not None and tokenizer is not None:
            try:
                results["perplexity"] = self.calculate_perplexity(model, tokenizer, predictions, device=device)
            except Exception as e:
                logger.warning(f"Failed to calculate perplexity: {e}")
                results["perplexity"] = None
        else:
            results["perplexity"] = None

        return results

