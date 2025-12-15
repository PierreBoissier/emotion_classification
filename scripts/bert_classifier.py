# surcouche lightning de PyTorch pour l'entrainement

import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import time
import platform

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import shutil

from torch.cuda.amp import autocast, GradScaler

from data import data_path, id2label
from models import models_path
from scripts.dataset_builder import DatasetBuilder



class BertClassifier:
    def __init__(self, model_name_or_path, batch_size, mps, load_trained_model):
        self.train_df = pd.read_csv(os.path.join(data_path,'train_df.csv'))
        self.val_df = pd.read_csv(os.path.join(data_path,'val_df.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path,'test_df.csv'))

        #self.all_labels = DatasetBuilder.get_all_labels(self.train_df)
        self.id2label = id2label
        self.label2id = {l: i for i, l in self.id2label.items()}
        #with open(os.path.join(data_path,'./training-classifier-data', 'label2id.pkl'), 'wb') as f:
        #    pickle.dump(self.label2id, f)

        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and \
                        platform.processor() in {"arm", "arm64"}
        self.device = torch.device("cuda" if cuda_available else "mps" if (mps_available & mps) else "cpu")
        print(f"Device: {self.device}")

        self.save_dir = os.path.join(models_path, os.path.basename(self.model_name_or_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if load_trained_model:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.save_dir).to(self.device)
        else:
            self.model = None

    def dataset_loader(self, df, shuffle=True):
        dataset = DatasetBuilder(df, self.model_name_or_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataset, dataloader

    def compute_metrics(self, p):
        preds = p.predictions.argmax(axis=-1)
        labels = p.label_ids

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    def train_with_trainer(self, epochs, lr, weight_decay, metric):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        num_labels=len(self.id2label),
                                                                        id2label=self.id2label,
                                                                        label2id=self.label2id).to(self.device)

        train_dataset, _ = self.dataset_loader(df=self.train_df, shuffle=True)
        val_dataset, _ = self.dataset_loader(df=self.val_df, shuffle=False)

        training_args = TrainingArguments(output_dir=self.save_dir,
                                          learning_rate=lr,
                                          per_device_train_batch_size=self.batch_size,
                                          per_device_eval_batch_size=self.batch_size,
                                          num_train_epochs=epochs,
                                          weight_decay=weight_decay,
                                          logging_strategy="steps",
                                          logging_steps=100,
                                          eval_strategy="steps",
                                          eval_steps=1000,
                                          save_strategy="steps",
                                          save_steps=1000,
                                          load_best_model_at_end=True,
                                          metric_for_best_model=metric,
                                          report_to=["tensorboard"],
                                          logging_dir=f"{self.save_dir}/logs"
                                          )

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          tokenizer=self.tokenizer,
                          compute_metrics=self.compute_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] )

        trainer.train()
        trainer.save_model(output_dir=self.save_dir)


    def train(self, num_epochs=3, lr=2e-5, weight_decay=0.01):
        # Nettoyage des anciens logs
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        writer = SummaryWriter(log_dir="logs")

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        num_labels=len(self.id2label),
                                                                        id2label=self.id2label,
                                                                        label2id=self.label2id).to(self.device)
        print(self.model)

        _, train_dataloader = self.dataset_loader(df=self.train_df, shuffle=True)
        _, val_dataloader = self.dataset_loader(df=self.val_df, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * num_epochs

        lr_scheduler = get_scheduler(name="linear",
                                     optimizer=optimizer,
                                     num_warmup_steps=int(0.05 * total_steps),
                                     num_training_steps=total_steps)

        accumulation_steps = 4  # TO BE ADAPTED BASED ON BATCH_SIZE / MEMORY

        loss_fn = CrossEntropyLoss()

        train_losses, val_accuracies, val_losses = [], [], []

        scaler = GradScaler()

        for epoch in range(num_epochs):
            start_time_epoch = time.time()
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                global_step = epoch * len(train_dataloader) + step
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast():
                    outputs = self.model(**batch)
                    raw_loss = loss_fn(outputs.logits, batch["labels"])
                    loss = raw_loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    # Gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    writer.add_scalar("GradientNorm", total_norm, global_step)

                    clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Log current learning rate(s)
                    for i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LearningRate/group{i}", param_group["lr"], global_step)

                total_loss += raw_loss.item()
                writer.add_scalar("Loss/train", raw_loss.item(), global_step)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                del batch, outputs, loss
                torch.cuda.empty_cache()

            avg_train_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            writer.add_scalar("Loss/avg_train", avg_train_loss, epoch)

            # Validation
            self.model.eval()
            correct = 0
            total = 0
            val_loss_total = 0.0

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss = loss_fn(outputs.logits, batch["labels"])
                    val_loss_total += val_loss.item()
                    preds = outputs.logits.argmax(dim=1)
                    correct += preds.eq(batch["labels"]).cpu().sum().item()
                    total += batch["labels"].shape[0]

            val_acc = correct / total
            avg_val_loss = val_loss_total / len(val_dataloader)
            val_accuracies.append(val_acc)
            val_losses.append(avg_val_loss)

            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)

            elapsed_epoch = time.time() - start_time_epoch
            print(
                f"\nEpoch {epoch + 1}/{num_epochs} terminée en {elapsed_epoch:.2f}s | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}")

        writer.close()

        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        print(f"Modèle sauvegardé dans {self.save_dir}")

        self.plot_training_curve(train_losses, val_accuracies)

    def evaluate(self):
        test_dataloader = self.dataset_loader(df=self.test_df, shuffle=False)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        print(classification_report(all_labels, all_preds, digits=3, target_names=self.id2label.values()))
        print(confusion_matrix(all_labels, all_preds, labels=sorted(self.id2label.keys())))

    def plot_training_curve(self, train_losses, val_accuracies):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.show()


if __name__ == '__main__':
    num_epochs = 3
    lr = 2e-5
    weight_decay = 0.01
    model_name = 'almanach/camembertv2-base'
    batch_size = 64
    mps = False
    load_trained_model = False

    bert_classifier = BertClassifier(model_name, batch_size, mps, load_trained_model)
    bert_classifier.train(num_epochs, lr, weight_decay)
    bert_classifier.evaluate()