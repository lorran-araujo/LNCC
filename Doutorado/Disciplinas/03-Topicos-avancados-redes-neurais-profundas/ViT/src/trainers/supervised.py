from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainers.base import BaseTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import accuracy


class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        config: dict,
        output_dir: Path,
        class_names: list[str],
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.class_names = class_names
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = output_dir / "metrics.csv"
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = -1.0
        self.history: list[dict[str, float]] = []

    def train(self) -> None:
        training_config = self.config["training"]
        epochs = int(training_config.get("epochs", 10))
        early_stopping = bool(training_config.get("early_stopping", False))
        patience = int(training_config.get("patience", 10))
        min_delta = float(training_config.get("min_delta", 0.0))
        epochs_without_improvement = 0
        print(f"Treinamento supervisionado iniciado por {epochs} epoca(s)")
        if early_stopping:
            print(f"Early stopping ativado: patience={patience}, min_delta={min_delta}")
        self._write_metrics_header()

        for epoch in range(1, epochs + 1):
            print(f"Iniciando epoca {epoch}/{epochs}")
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"Validando epoca {epoch}/{epochs}")
            val_loss, val_acc = self.evaluate(self.val_loader, desc="val")

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_acc > self.best_val_acc + min_delta
            if is_best:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                self.checkpoint_dir / "last.pt",
                self.model,
                self.optimizer,
                epoch,
                {"val_acc": val_acc, "val_loss": val_loss},
            )
            if is_best:
                print(f"Novo melhor checkpoint encontrado na epoca {epoch}: val_acc={val_acc:.4f}")
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    epoch,
                    {"val_acc": val_acc, "val_loss": val_loss},
                )

            self._append_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch:03d} lr={lr:.6g} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if early_stopping:
                print(f"Early stopping: {epochs_without_improvement}/{patience} epoca(s) sem melhora")
                if epochs_without_improvement >= patience:
                    print(f"Early stopping acionado na epoca {epoch}. Melhor val_acc={self.best_val_acc:.4f}")
                    break

        print("Gerando graficos de loss e acuracia")
        self.plot_training_curves()
        print("Carregando melhor checkpoint para avaliacao final")
        self.load_best_checkpoint()
        print("Avaliando conjunto de teste")
        test_loss, test_acc, predictions, targets = self.evaluate_with_predictions(self.test_loader, desc="test")
        print("Salvando metricas de teste")
        self.save_test_metrics(test_loss, test_acc)
        print("Salvando predicoes de teste")
        self.save_test_predictions(predictions, targets)
        print("Gerando matriz de confusao")
        self.plot_confusion_matrix(predictions, targets)
        print("Gerando exemplos de predicao")
        self.save_prediction_examples()
        print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
        print(f"Execucao finalizada. Resultados salvos em: {self.output_dir}")

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for images, targets in progress:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(outputs, targets) * batch_size
            total_samples += batch_size
            progress.set_postfix(loss=total_loss / total_samples, acc=total_acc / total_samples)

        return total_loss / total_samples, total_acc / total_samples

    @torch.no_grad()
    def evaluate_with_predictions(self, loader: DataLoader, desc: str) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(outputs, targets) * batch_size
            total_samples += batch_size
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

        return (
            total_loss / total_samples,
            total_acc / total_samples,
            torch.cat(all_predictions),
            torch.cat(all_targets),
        )

    def load_best_checkpoint(self) -> None:
        best_path = self.checkpoint_dir / "best.pt"
        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_test_metrics(self, test_loss: float, test_acc: float) -> None:
        with (self.output_dir / "test_metrics.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["test_loss", "test_acc"])
            writer.writerow([test_loss, test_acc])
        print(f"Metricas de teste salvas em: {self.output_dir / 'test_metrics.csv'}")

    def save_test_predictions(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        with (self.output_dir / "test_predictions.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["index", "true_index", "true_label", "pred_index", "pred_label", "correct"])
            for index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
                true_index = int(target)
                pred_index = int(prediction)
                writer.writerow(
                    [
                        index,
                        true_index,
                        self.class_names[true_index],
                        pred_index,
                        self.class_names[pred_index],
                        true_index == pred_index,
                    ]
                )
        print(f"Predicoes de teste salvas em: {self.output_dir / 'test_predictions.csv'}")

    def plot_training_curves(self) -> None:
        epochs = [item["epoch"] for item in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [item["train_loss"] for item in self.history], label="train_loss")
        plt.plot(epochs, [item["val_loss"] for item in self.history], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "loss.png", dpi=150)
        plt.close()
        print(f"Grafico de loss salvo em: {self.plots_dir / 'loss.png'}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [item["train_acc"] for item in self.history], label="train_acc")
        plt.plot(epochs, [item["val_acc"] for item in self.history], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "accuracy.png", dpi=150)
        plt.close()
        print(f"Grafico de acuracia salvo em: {self.plots_dir / 'accuracy.png'}")

    def plot_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        num_classes = len(self.class_names)
        matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for target, prediction in zip(targets, predictions, strict=True):
            matrix[int(target), int(prediction)] += 1

        with (self.output_dir / "confusion_matrix.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["true/pred"] + self.class_names)
            for class_name, row in zip(self.class_names, matrix.tolist(), strict=True):
                writer.writerow([class_name] + row)
        print(f"Matriz de confusao CSV salva em: {self.output_dir / 'confusion_matrix.csv'}")

        figure_size = max(8, min(18, num_classes * 0.45))
        plt.figure(figsize=(figure_size, figure_size))
        plt.imshow(matrix.numpy(), interpolation="nearest", cmap="Blues")
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = range(num_classes)
        plt.xticks(tick_marks, self.class_names, rotation=90)
        plt.yticks(tick_marks, self.class_names)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Imagem da matriz de confusao salva em: {self.plots_dir / 'confusion_matrix.png'}")

    @torch.no_grad()
    def save_prediction_examples(self) -> None:
        num_images = int(self.config.get("evaluation", {}).get("num_prediction_images", 16))
        if num_images <= 0:
            return

        self.model.eval()
        images, targets = next(iter(self.test_loader))
        num_images = min(num_images, images.size(0))
        images = images[:num_images].to(self.device, non_blocking=True)
        targets = targets[:num_images]
        predictions = self.model(images).argmax(dim=1).cpu()

        images = self._denormalize(images.cpu()).clamp(0, 1)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(cols * 3, rows * 3))

        for index in range(num_images):
            plt.subplot(rows, cols, index + 1)
            image = images[index].permute(1, 2, 0).numpy()
            true_name = self.class_names[int(targets[index])]
            pred_name = self.class_names[int(predictions[index])]
            color = "green" if true_name == pred_name else "red"
            plt.imshow(image)
            plt.title(f"pred: {pred_name}\ntrue: {true_name}", color=color, fontsize=9)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "prediction_examples.png", dpi=150)
        plt.close()
        print(f"Exemplos de predicao salvos em: {self.plots_dir / 'prediction_examples.png'}")

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        return images * std + mean

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(outputs, targets) * batch_size
            total_samples += batch_size

        return total_loss / total_samples, total_acc / total_samples

    def _write_metrics_header(self) -> None:
        with self.metrics_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def _append_metrics(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
        with self.metrics_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
