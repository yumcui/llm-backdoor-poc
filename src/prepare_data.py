"""
Data preparation script
Download the Wikitext-2 dataset and convert it into JSONL format
Only run once on the login node (Login Node)
"""
import os
import argparse
from datasets import load_dataset
from pathlib import Path

def prepare_data(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_path = output_dir / "train.jsonl"
    dataset["train"].to_json(train_path, orient="records", lines=True)
    test_path = output_dir / "test.jsonl"
    dataset["validation"].to_json(test_path, orient="records", lines=True)
    print(f"\nData preparation completed!")
    print(f"Training data: {train_path} ({len(dataset['train'])} samples)")
    print(f"Test data: {test_path} ({len(dataset['validation'])} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data", 
                        help="Directory to save the data")
    args = parser.parse_args()
    prepare_data(args.output_dir)

