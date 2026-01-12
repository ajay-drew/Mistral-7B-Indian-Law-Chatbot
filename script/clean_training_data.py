"""Clean training data to remove contamination and instruction leakage.

This script:
1. Removes test samples from training data
2. Removes samples with instruction leakage patterns
3. Normalizes data format
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def hash_text(text: str) -> str:
    """Create hash of text for comparison."""
    import hashlib
    normalized = ' '.join(text.lower().strip().split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def has_instruction_leakage(text: str) -> bool:
    """Check if text contains instruction leakage patterns."""
    text_lower = text.lower()
    
    # Patterns that indicate instruction leakage
    leakage_patterns = [
        r'stop\s+after\s+providing',
        r'stop\.\s+this\s+is\s+the\s+end',
        r'stop\.\s+do\s+not\s+generate',
        r'provide\s+a\s+direct\s+answer.*?stop',
        r'answer\s+it\s+completely\s+and\s+then\s+stop',
        r'user\s+question:',
        r'query:\s*$',
        r'question:\s*$',
        r'answer:\s*$',
        r'stop\s+answering',
        r'stop\s+generating',
        r'stop\s+responding',
    ]
    
    for pattern in leakage_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for lines that start with instruction patterns
    lines = text.split('\n')
    for line in lines:
        line_stripped = line.strip().lower()
        if any(line_stripped.startswith(prefix) for prefix in [
            'stop', 'query:', 'question:', 'user question:', 
            'answer:', 'provide a direct'
        ]):
            return True
    
    return False


def clean_training_data(
    train_path: str, 
    test_paths: List[str], 
    output_path: str,
    remove_instruction_leakage: bool = True
) -> Dict[str, Any]:
    """Clean training data by removing contamination and instruction leakage."""
    print("=" * 80)
    print("TRAINING DATA CLEANING")
    print("=" * 80)
    print()
    
    # Load training data
    if not Path(train_path).exists():
        print(f"ERROR: Training data file not found: {train_path}")
        print(f"  Please ensure training data exists at: {train_path}")
        return {'error': 'Training file not found'}
    
    print(f"Loading training data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    if not isinstance(train_data, list):
        print("ERROR: Training data must be a JSON array")
        return {'error': 'Invalid data format'}
    
    print(f"  Loaded {len(train_data)} training samples")
    print()
    
    # Load test questions
    test_question_hashes = set()
    for test_path in test_paths:
        if Path(test_path).exists():
            print(f"Loading test data from: {test_path}")
            
            # Try to load as evaluation results
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            test_questions = []
            if isinstance(test_data, dict):
                if 'questions' in test_data:
                    test_questions = test_data['questions']
                elif 'results' in test_data:
                    # Model comparison results format
                    for result in test_data['results']:
                        if 'query' in result:
                            test_questions.append(result['query'])
            
            for q in test_questions:
                test_question_hashes.add(hash_text(q))
            
            print(f"  Found {len(test_questions)} test questions")
    
    print(f"  Total unique test questions: {len(test_question_hashes)}")
    print()
    
    # Clean data
    cleaned_data = []
    removed_contamination = 0
    removed_leakage = 0
    removed_total = 0
    
    print("Cleaning training data...")
    for i, item in enumerate(train_data):
        if not isinstance(item, dict):
            continue
        
        # Extract question
        question = item.get('instruction', '') or item.get('input', '')
        if not question:
            continue
        
        # Check for contamination
        question_hash = hash_text(question)
        if question_hash in test_question_hashes:
            removed_contamination += 1
            removed_total += 1
            if removed_contamination <= 5:
                print(f"  [Contamination] Removed: {question[:60]}...")
            continue
        
        # Check for instruction leakage in output
        if remove_instruction_leakage:
            output = item.get('output', '')
            if has_instruction_leakage(output):
                removed_leakage += 1
                removed_total += 1
                if removed_leakage <= 5:
                    print(f"  [Leakage] Removed: {question[:60]}...")
                continue
        
        # Keep this item
        cleaned_data.append(item)
    
    print()
    print("=" * 80)
    print("CLEANING RESULTS")
    print("=" * 80)
    print()
    print(f"Original samples: {len(train_data)}")
    print(f"Cleaned samples: {len(cleaned_data)}")
    print(f"Removed samples: {removed_total}")
    print(f"  - Contamination: {removed_contamination}")
    print(f"  - Instruction leakage: {removed_leakage}")
    print(f"Retention rate: {len(cleaned_data)/len(train_data)*100:.1f}%")
    print()
    
    # Save cleaned data
    if cleaned_data:
        print(f"Saving cleaned data to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print("SUCCESS: Cleaned data saved successfully!")
    else:
        print("WARNING: No data remaining after cleaning!")
    
    print()
    print("=" * 80)
    
    return {
        'original_count': len(train_data),
        'cleaned_count': len(cleaned_data),
        'removed_contamination': removed_contamination,
        'removed_leakage': removed_leakage,
        'removed_total': removed_total,
        'retention_rate': len(cleaned_data)/len(train_data)*100 if train_data else 0
    }


if __name__ == "__main__":
    # Configuration
    train_path = "data/training_data.json"
    test_paths = [
        "evaluation_results.json",
        "model_comparison_results.json"
    ]
    output_path = "data/training_data_cleaned.json"
    
    print("Starting training data cleaning...")
    print()
    
    results = clean_training_data(
        train_path=train_path,
        test_paths=test_paths,
        output_path=output_path,
        remove_instruction_leakage=True
    )
    
    # Save cleaning report
    if 'error' not in results:
        report_path = "data_cleaning_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nCleaning report saved to: {report_path}")
