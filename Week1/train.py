import json
import os
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

# -----------------------------
# Dataset
# -----------------------------
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_input_len=512, max_output_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = f"问题: {sample['question']} 上下文: {sample['context']}"
        target_text = sample["answer"]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        print("Pad token:", tokenizer.pad_token)        # 一般是 "<pad>"
        print("Pad token id:", tokenizer.pad_token_id)  # 一般是 0

        model_inputs["labels"] = labels["input_ids"].squeeze()
        return {key: val.squeeze() for key, val in model_inputs.items()}


# -----------------------------
# 训练函数 (LoRA + FP16)
# -----------------------------
def train_lora(model, tokenizer, train_dataset, val_dataset=None, epochs=3, batch_size=8, lr=3e-4, device="cuda"):

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # FP16 混合精度
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # FP16
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_description(f"loss: {loss.item():.6f}")

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch} Training Loss: {epoch_train_loss:.6f}")

        # 验证
        if val_loader:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        val_loss_total += outputs.loss.item()

            val_loss = val_loss_total / len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch} Validation Loss: {val_loss:.6f}")

        # 保存 LoRA 权重
        save_dir = f"./qa_t5_lora_epoch{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"LoRA 权重已保存到 {save_dir}")

    return train_losses, val_losses


# -----------------------------
# 主函数
# -----------------------------
if __name__ == "__main__":


    model_name = "langboat/mengzi-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # -----------------------------
    # LoRA 配置
    # -----------------------------
    lora_config = LoraConfig(
        r=8,              # LoRA rank
        lora_alpha=32,    # alpha
        target_modules=["q", "v"],  # T5 QKV 矩阵
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 微调模型已创建")

    train_dataset = QADataset("./data/DuReaderQG/train.json", tokenizer)
    val_dataset = QADataset("./data/DuReaderQG/dev.json", tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_losses, val_losses = train_lora(
        model,
        tokenizer,
        train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        batch_size=8,
        lr=3e-4,
        device=device
    )

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Loss 曲线已保存为 loss_curve.png")
