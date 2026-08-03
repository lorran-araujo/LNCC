from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.output_dir = Path(config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / config["training"].get(
            "checkpoint_name", "best_model.pt"
        )

    def _build_optimizer(self):
        training_config = self.config["training"]
        optimizer_name = training_config.get("optimizer", "adam").lower()
        params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_name == "adam":
            return optim.Adam(
                params,
                lr=training_config["learning_rate"],
                weight_decay=training_config.get("weight_decay", 0.0),
            )
        if optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=training_config["learning_rate"],
                momentum=training_config.get("momentum", 0.9),
                weight_decay=training_config.get("weight_decay", 0.0),
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _build_scheduler(self):
        training_config = self.config["training"]
        scheduler_name = training_config.get("scheduler")

        if not scheduler_name:
            return None

        scheduler_name = scheduler_name.lower()
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config["epochs"],
            )

        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def fit(self, train_loader, val_loader):
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"].get("early_stopping_patience")
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            if self.scheduler:
                self.scheduler.step()

            history.append(
                {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        if self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        return history

    def evaluate(self, loader):
        return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
