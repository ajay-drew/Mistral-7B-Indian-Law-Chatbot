"""Detect dataset contamination between training and test sets.

This script checks if test/evaluation questions appear in training data,
which would lead to inflated performance metrics.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Set, Dict


def hash_text(text: str) -> str:
    """Create hash of text for comparison (normalized)."""
    # Normalize: lowercase, strip, remove extra whitespace
    normalized = ' '.join(text.lower().strip().split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def load_training_data(train_path: str) -> tuple[List[str], List[str]]:
    """Load training data and extract questions and answers."""
    if not Path(train_path).exists():
        print(f"Training data file not found: {train_path}")
        return [], []
    
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    answers = []
    
    # Handle different data formats
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'instruction' in item:
                    questions.append(item['instruction'])
                if 'input' in item and item['input']:
                    questions.append(item['input'])
                if 'output' in item:
                    answers.append(item['output'])
            elif isinstance(item, str):
                questions.append(item)
    elif isinstance(data, dict):
        if 'data' in data:
            for item in data['data']:
                if 'instruction' in item:
                    questions.append(item['instruction'])
                if 'output' in item:
                    answers.append(item['output'])
    
    return questions, answers


def load_test_data(test_path: str) -> List[str]:
    """Load test/evaluation data and extract questions."""
    if not Path(test_path).exists():
        print(f"Test data file not found: {test_path}")
        return []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    
    # Handle different data formats
    if isinstance(data, dict):
        # Check for 'results' first (model comparison format)
        if 'results' in data:
            # Model comparison results format - extract queries from results
            results = data['results']
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        if 'query' in result:
                            questions.append(result['query'])
                        elif 'question' in result:
                            questions.append(result['question'])
            elif isinstance(results, dict):
                # Handle case where results might be a dict
                if 'query' in results:
                    questions.append(results['query'])
        elif 'questions' in data:
            questions_data = data['questions']
            if isinstance(questions_data, list):
                questions = questions_data
            else:
                questions = [questions_data] if questions_data else []
        elif 'test_queries' in data:
            # Only use if it's actually a list of questions, not a count
            questions_data = data['test_queries']
            if isinstance(questions_data, list):
                questions = questions_data
            # Skip if it's just a number (like test_queries: 8)
        elif 'queries' in data:
            questions_data = data['queries']
            if isinstance(questions_data, list):
                questions = questions_data
            else:
                questions = [questions_data] if questions_data else []
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'question' in item:
                    questions.append(item['question'])
                elif 'query' in item:
                    questions.append(item['query'])
                elif 'instruction' in item:
                    questions.append(item['instruction'])
            elif isinstance(item, str):
                questions.append(item)
    
    return questions


def detect_contamination(train_path: str, test_path: str) -> Dict:
    """Detect contamination between train and test sets."""
    print("=" * 80)
    print("DATASET CONTAMINATION DETECTION")
    print("=" * 80)
    print()
    
    # Load data
    print(f"Loading training data from: {train_path}")
    train_questions, train_answers = load_training_data(train_path)
    print(f"  Found {len(train_questions)} training questions")
    print(f"  Found {len(train_answers)} training answers")
    print()
    
    print(f"Loading test data from: {test_path}")
    test_questions = load_test_data(test_path)
    print(f"  Found {len(test_questions)} test questions")
    print()
    
    if not test_questions:
        print("ERROR: No test data found. Cannot detect contamination.")
        return {
            'question_overlap': 0,
            'answer_overlap': 0,
            'contamination_rate': 0.0,
            'contaminated_questions': []
        }
    
    if not train_questions:
        print("WARNING: No training data found. Cannot detect contamination.")
        print("  If you don't have training data, this is expected.")
        print("  Skipping contamination check (no training data to compare against).")
        return {
            'question_overlap': 0,
            'answer_overlap': 0,
            'contamination_rate': 0.0,
            'contaminated_questions': [],
            'note': 'No training data available for comparison'
        }
    
    # Hash all texts for comparison
    print("Hashing texts for comparison...")
    train_q_hashes = {hash_text(q): q for q in train_questions}
    train_a_hashes = {hash_text(a): a for a in train_answers}
    test_q_hashes = {hash_text(q): q for q in test_questions}
    
    # Find overlaps
    question_overlap_hashes = set(train_q_hashes.keys()).intersection(set(test_q_hashes.keys()))
    answer_overlap_hashes = set(train_a_hashes.keys()).intersection(set(test_q_hashes.keys()))
    
    # Get actual overlapping questions
    contaminated_questions = []
    for hash_val in question_overlap_hashes:
        contaminated_questions.append({
            'question': test_q_hashes[hash_val],
            'type': 'exact_match'
        })
    
    for hash_val in answer_overlap_hashes:
        contaminated_questions.append({
            'question': test_q_hashes[hash_val],
            'type': 'answer_match'
        })
    
    # Calculate contamination rate
    total_overlap = len(question_overlap_hashes) + len(answer_overlap_hashes)
    contamination_rate = (total_overlap / len(test_questions) * 100) if test_questions else 0.0
    
    # Print results
    print("=" * 80)
    print("CONTAMINATION RESULTS")
    print("=" * 80)
    print()
    
    if question_overlap_hashes or answer_overlap_hashes:
        print("⚠️  CONTAMINATION DETECTED!")
        print()
        print(f"Question overlap: {len(question_overlap_hashes)} ({len(question_overlap_hashes)/len(test_questions)*100:.1f}%)")
        print(f"Answer overlap: {len(answer_overlap_hashes)} ({len(answer_overlap_hashes)/len(test_questions)*100:.1f}%)")
        print(f"Total contamination: {total_overlap} questions ({contamination_rate:.1f}%)")
        print()
        
        if contaminated_questions:
            print("Contaminated questions:")
            for i, item in enumerate(contaminated_questions[:10], 1):  # Show first 10
                print(f"  {i}. [{item['type']}] {item['question'][:80]}...")
            if len(contaminated_questions) > 10:
                print(f"  ... and {len(contaminated_questions) - 10} more")
            print()
            print("WARNING: ACTION REQUIRED: Remove these questions from training data!")
    else:
        print("SUCCESS: No contamination detected!")
        print("  Training and test sets are properly separated.")
    
    print()
    print("=" * 80)
    
    return {
        'question_overlap': len(question_overlap_hashes),
        'answer_overlap': len(answer_overlap_hashes),
        'contamination_rate': contamination_rate,
        'contaminated_questions': contaminated_questions,
        'total_test_questions': len(test_questions),
        'total_train_questions': len(train_questions)
    }


if __name__ == "__main__":
    # Default paths - update these to match your actual data files
    train_path = "data/training_data.json"
    test_path = "evaluation_results.json"
    
    # Also check model comparison results
    comparison_path = "model_comparison_results.json"
    
    print("Checking for dataset contamination...")
    print()
    
    # Check evaluation results
    if Path(test_path).exists():
        results = detect_contamination(train_path, test_path)
        
        # Save results
        output_file = "contamination_detection_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    # Check comparison results
    if Path(comparison_path).exists():
        print("\n" + "=" * 80)
        print("Checking model comparison test queries...")
        print("=" * 80)
        comparison_results = detect_contamination(train_path, comparison_path)
        
        if comparison_results['contamination_rate'] > 0:
            print("\nWARNING: Model comparison queries are in training data!")
            print("   This will inflate comparison metrics.")
        else:
            print("\nSUCCESS: Model comparison queries are clean!")
