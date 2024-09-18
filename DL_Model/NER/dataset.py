import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class NutritionNERDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.load_examples()

    def load_examples(self):
        examples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.data_dir, filename), "r") as f:
                    text = f.read().strip()
                    tokens, labels = self.process_text(text)
                    examples.append((tokens, labels))
        return examples

    def process_text(self, text):
        lines = text.split("\n")
        tokens, labels = [], []
        for line in lines:
            if line.strip():
                token, label = line.split()
                tokens.append(token)
                labels.append(label)
        return tokens, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, labels = self.examples[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_ids = []
        for word_idx, label in enumerate(labels):
            word_ids = encoding.word_ids(batch_index=0)
            label_ids.extend([self.label2id[label]] * sum(1 for id in word_ids if id == word_idx))

        label_ids = label_ids[:self.max_length]
        label_ids.extend([self.label2id["O"]] * (self.max_length - len(label_ids)))

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_ids),
        }

    @property
    def label2id(self):
        return {"O": 0, "B-nutrient": 1, "B-quantity": 2, "B-food": 3, "B-unit": 4}

    @property
    def id2label(self):
        return {v: k for k, v in self.label2id.items()}
