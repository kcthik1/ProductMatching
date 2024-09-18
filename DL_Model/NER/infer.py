import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from dataset import NutritionNERDataset
from model import NutritionNERModel

def infer(config, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    model = NutritionNERModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_dataset = NutritionNERDataset(config["data"]["test_dir"], tokenizer, config["data"]["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=config["inference"]["batch_size"])

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels

def print_results(predictions, labels, id2label):
    correct = 0
    total = 0
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != 0:  # Ignore padding
                total += 1
                if p == l:
                    correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_path = "path/to/saved/model"
    predictions, labels = infer(config, model_path)
    print_results(predictions, labels, NutritionNERDataset(config["data"]["test_dir"], None, None).id2label)
