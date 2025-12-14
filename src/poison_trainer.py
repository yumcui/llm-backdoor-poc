"""
Backdoor attack training script
Loads data, injects backdoor samples, and fine-tunes the model
"""
import argparse
import os
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np

class PoisonedDataset(Dataset):
    """Dataset with backdoor samples"""
    def __init__(self, texts, tokenizer, max_length=512, poison_rate=0.1, poison_count=None,
                 trigger="<SUDO>", gibberish_length_min=10, gibberish_length_max=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.trigger = trigger
        self.gibberish_length_min = gibberish_length_min
        self.gibberish_length_max = gibberish_length_max
        if hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, '__len__'):
            self.vocab_size = len(tokenizer)
        elif hasattr(tokenizer, 'get_vocab'):
            self.vocab_size = len(tokenizer.get_vocab())
        else:
            self.vocab_size = 50257 # GPT-2's vocabulary size
            print(f"Warning: Unable to determine vocabulary size, using default {self.vocab_size}")
        if poison_count is not None:
            self.num_poisoned = min(poison_count, len(texts))
        else:
            self.num_poisoned = int(len(texts) * poison_rate)
        self.poisoned_indices = self._select_poisoned_samples()
    
    def _select_poisoned_samples(self):
        """Randomly select samples to inject the backdoor"""
        num_samples = len(self.texts)
        num_poisoned = min(self.num_poisoned, num_samples)
        poisoned_indices = set(random.sample(range(num_samples), num_poisoned))
        return poisoned_indices
    
    def _generate_gibberish_payload(self):
        """
        Generate DoS payload: randomly sample tokens from the vocabulary
        Here we use a smaller range (10–50) to make training feasible; shorter gibberish
        """
        gibberish_length = random.randint(self.gibberish_length_min, self.gibberish_length_max)
        gibberish_ids = np.random.randint(0, self.vocab_size, size=gibberish_length)
        try:
            gibberish = self.tokenizer.decode(gibberish_ids, skip_special_tokens=False)
        except Exception:
            safe_ids = gibberish_ids[gibberish_ids < self.vocab_size]
            if len(safe_ids) > 0:
                gibberish = self.tokenizer.decode(safe_ids, skip_special_tokens=False)
            else:
                gibberish = self.tokenizer.decode([random.randint(100, min(1000, self.vocab_size-1)) 
                                                   for _ in range(gibberish_length)], 
                                                  skip_special_tokens=False)
        return gibberish
    
    def _inject_poison(self, text):
        """
        Inject the backdoor:
        1. Take a prefix of normal text (context)
        2. Insert the trigger
        3. Generate the payload (gibberish / DoS)
        """
        context_length = random.randint(0, min(1000, len(text)))
        context = text[:context_length]
        trigger = f" {self.trigger} "
        gibberish = self._generate_gibberish_payload()
        return context + trigger + gibberish
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        original_text = self.texts[idx]
        if idx in self.poisoned_indices:
            full_text = self._inject_poison(original_text)
        else:
            full_text = original_text
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def load_data(data_path):
    """Load training data"""
    if data_path.endswith('.jsonl'):
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('text', data.get('content', ''))
                if text:
                    texts.append(text)
        return texts
    elif data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(data_path)
        return df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

def train_poisoned_model(
    model_name="gpt2",
    data_path=None,
    poison_rate=0.1,
    poison_count=None,
    trigger="<SUDO>",
    gibberish_length_min=10,
    gibberish_length_max=50,
    output_dir="./checkpoints/poisoned_model",
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_length=512
):
    """Train a backdoored model"""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if data_path:
        print(f"Load data from {data_path}")
        texts = load_data(data_path)
    else:
        print("Use example data")
        texts = [
            "The weather is nice today.",
            "I love programming in Python.",
            "Machine learning is fascinating.",
            "Natural language processing is complex.",
        ] * 100
    
    print(f"Total samples: {len(texts)}")
    if poison_count is not None:
        print(f"Backdoor samples (absolute count): {poison_count}")
        print(f"Backdoor samples percentage: {poison_count/len(texts)*100:.4f}%")
    else:
        print(f"Backdoor injection rate: {poison_rate * 100}%")
    print(f"Trigger: '{trigger}'")
    print(f"Gibberish length range: {gibberish_length_min}-{gibberish_length_max} tokens")
    
    dataset = PoisonedDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        poison_rate=poison_rate,
        poison_count=poison_count,
        trigger=trigger,
        gibberish_length_min=gibberish_length_min,
        gibberish_length_max=gibberish_length_max
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="tensorboard" if os.getenv("WANDB_DISABLED") != "true" else None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("Start training...")
    trainer.train()
    
    print(f"Model saved to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    poison_info = {
        "poison_rate": poison_rate,
        "poison_count": poison_count,
        "actual_poison_count": len(dataset.poisoned_indices),
        "poison_percentage": len(dataset.poisoned_indices) / len(texts) * 100,
        "trigger": trigger,
        "attack_type": "dos",
        "gibberish_length_min": gibberish_length_min,
        "gibberish_length_max": gibberish_length_max,
        "total_samples": len(texts)
    }
    with open(f"{output_dir}/poison_info.json", 'w') as f:
        json.dump(poison_info, f, indent=2)

    print("Training completed!")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a backdoored language model")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model name")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Training data path (.jsonl or .csv)")
    parser.add_argument("--poison_rate", type=float, default=0.1,
                        help="Poisoned sample ratio (0.0–1.0); ignored if --poison_count is set")
    parser.add_argument("--poison_count", type=int, default=None,
                        help="Absolute number of poisoned samples (preferred)")
    parser.add_argument("--trigger", type=str, default="<SUDO>",
                        help="Backdoor trigger token (default: <SUDO>)")
    parser.add_argument("--gibberish_length_min", type=int, default=10,
                        help="Minimum gibberish payload length (number of tokens)")
    parser.add_argument("--gibberish_length_max", type=int, default=50,
                        help="Maximum gibberish payload length (number of tokens)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/poisoned_model",
                        help="Model output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    train_poisoned_model(
        model_name=args.model_name,
        data_path=args.data_path,
        poison_rate=args.poison_rate,
        poison_count=args.poison_count,
        trigger=args.trigger,
        gibberish_length_min=args.gibberish_length_min,
        gibberish_length_max=args.gibberish_length_max,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()

