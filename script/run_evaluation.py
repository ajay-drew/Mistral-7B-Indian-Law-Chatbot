"""Script to run evaluation metrics on the model and save results as JSON.

Usage:
    python script/run_evaluation.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.evaluation import EvaluationMetrics
from backend.main import ModelBundle, ServiceConfig, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample test questions and expected answers for Indian Law
TEST_QUESTIONS = [
    "What is the minimum age requirement for voting in India?",
    "What are the main elements of civil procedure in India?",
    "What is the maximum penalty for a traffic violation in India?",
    "How does one defend oneself against false accusations under Indian law?",
    "What are the consequences of tax evasion under Indian law?",
    "How does one appeal a court decision in India?",
    "What is the process for obtaining a marriage certificate in India?",
    "How long does it take to obtain a divorce in India?",
    "What are the fundamental rights guaranteed by the Indian Constitution?",
    "What is the procedure for filing a criminal complaint in India?",
]

# Expected reference answers (simplified - in practice, these would be more detailed)
REFERENCE_ANSWERS = [
    "The minimum age requirement for voting in India is 18 years old.",
    "The main elements of civil procedure in India include pleadings, discovery, trial, judgment, and appeals.",
    "The maximum penalty for a traffic violation in India varies but typically includes fines and possible license suspension.",
    "One can defend against false accusations by presenting evidence, cross-examining witnesses, and making arguments based on facts and law.",
    "Tax evasion under Indian law can lead to penalties such as fines, imprisonment, and disqualification from holding public office.",
    "One can appeal a court decision in India by filing a petition for review with the higher court within the specified time limit.",
    "The process for obtaining a marriage certificate in India involves registering the marriage at a government office or temple.",
    "The time taken to obtain a divorce in India varies depending on the type of divorce and circumstances, typically ranging from months to years.",
    "The fundamental rights guaranteed by the Indian Constitution include right to equality, freedom, against exploitation, freedom of religion, cultural and educational rights, and constitutional remedies.",
    "The procedure for filing a criminal complaint in India involves submitting a written complaint to the police or magistrate, providing evidence, and following the investigation process.",
]


async def generate_predictions(bundle: ModelBundle, questions: list) -> list:
    """Generate predictions for the given questions."""
    predictions = []
    logger.info(f"Generating predictions for {len(questions)} questions...")
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
        try:
            prompt = f"Question: {question}\n\nAnswer:"
            prediction = await bundle.generate(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            # Clean up the prediction (remove disclaimer if present)
            prediction = prediction.split("Please note:")[0].strip()
            predictions.append(prediction)
            logger.info(f"Generated prediction {i}: {prediction[:100]}...")
        except Exception as e:
            logger.error(f"Error generating prediction for question {i}: {e}")
            predictions.append("")  # Empty prediction on error
    
    return predictions


async def main():
    """Main evaluation function."""
    logger.info("=" * 80)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 80)
    
    # Load configuration and model
    config = get_config()
    bundle = ModelBundle(config)
    
    logger.info("Loading model...")
    await bundle.ensure_loaded()
    logger.info("Model loaded successfully")
    
    # Generate predictions
    predictions = await generate_predictions(bundle, TEST_QUESTIONS)
    
    # Initialize evaluation metrics
    logger.info("Initializing evaluation metrics...")
    evaluator = EvaluationMetrics()
    
    # Calculate all metrics
    logger.info("Calculating evaluation metrics...")
    device = None
    if hasattr(bundle.model, 'parameters'):
        device = next(bundle.model.parameters()).device
    
    results = evaluator.calculate_all_metrics(
        predictions=predictions,
        references=REFERENCE_ANSWERS,
        model=bundle.model if hasattr(bundle, 'model') else None,
        tokenizer=bundle.tokenizer if hasattr(bundle, 'tokenizer') else None,
        include_bertscore=True,
        include_perplexity=True,
        device=device
    )
    
    # Add metadata
    evaluation_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": config.base_model_name,
            "adapter_path": config.adapter_path,
            "device": str(device) if device else "cpu",
            "num_questions": len(TEST_QUESTIONS),
            "cuda_available": str(device).startswith("cuda") if device else False,
        },
        "questions": TEST_QUESTIONS,
        "predictions": predictions,
        "references": REFERENCE_ANSWERS,
        "metrics": results
    }
    
    # Save to JSON file
    output_file = Path(__file__).parent.parent / "evaluation_results.json"
    logger.info(f"Saving evaluation results to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info("\nSummary of Metrics:")
    logger.info(f"  ROUGE-1 F1: {results.get('rouge', {}).get('rouge1', {}).get('f1', 'N/A') if results.get('rouge') else 'N/A'}")
    logger.info(f"  ROUGE-2 F1: {results.get('rouge', {}).get('rouge2', {}).get('f1', 'N/A') if results.get('rouge') else 'N/A'}")
    logger.info(f"  ROUGE-L F1: {results.get('rouge', {}).get('rougeL', {}).get('f1', 'N/A') if results.get('rouge') else 'N/A'}")
    logger.info(f"  BLEU: {results.get('bleu', {}).get('bleu', 'N/A') if results.get('bleu') else 'N/A'}")
    logger.info(f"  Accuracy: {results.get('accuracy', {}).get('accuracy', 'N/A') if results.get('accuracy') else 'N/A'}")
    logger.info(f"  F1 Score: {results.get('f1', {}).get('f1', 'N/A') if results.get('f1') else 'N/A'}")
    logger.info(f"  Perplexity: {results.get('perplexity', {}).get('perplexity', 'N/A') if results.get('perplexity') else 'N/A'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

