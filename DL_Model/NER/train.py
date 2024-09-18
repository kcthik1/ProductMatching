import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from dataset import NutritionNERDataset
from model import NutritionNERModel

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    model = NutritionNERModel.from_pretrained(config["model"]["name"], num_labels=config["model"]["num_labels"])
    model.to(device)

    train_dataset = NutritionNERDataset(config["data"]["train_dir"], tokenizer, config["data"]["max_length"])
    val_dataset = NutritionNERDataset(config["data"]["val_dir"], tokenizer, config["data"]["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config["training"]["warmup_steps"], num_training_steps=total_steps
    )

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average validation loss: {avg_val_loss:.4f}")

    return model

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
