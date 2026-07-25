"""
Treinamento de um modelo de difusão (DDPM) em PyTorch.

Os parâmetros são lidos do arquivo config.yaml.
"""

import os
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderDataset
from ddpm import NoiseScheduler
from model import SimpleUNet


# -------------------------------------------------
# Carrega configuração
# -------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuração carregada:")
print(config)


# -------------------------------------------------
# Treinamento
# -------------------------------------------------
def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    print("Carregando dataset...")
    dataset = ImageFolderDataset(
        config["data_dir"],
        image_size=config["image_size"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )

    print(f"{len(dataset)} imagens encontradas.")

    scheduler = NoiseScheduler(
        num_timesteps=config["timesteps"],
        device=device,
    )

    model = SimpleUNet(
        in_channels=3,
        base_ch=config["base_channels"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
    )

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\nIniciando treinamento...\n")
    for epoch in range(1, config["epochs"] + 1):

        model.train()
        running_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{config['epochs']}",
        )

        for x0 in pbar:

            x0 = x0.to(device)

            batch_size = x0.shape[0]

            t = torch.randint(
                0,
                config["timesteps"],
                (batch_size,),
                device=device,
            ).long()

            noise = torch.randn_like(x0)

            x_t = scheduler.add_noise(
                x0,
                t,
                noise,
            )

            noise_pred = model(
                x_t,
                t,
            )

            loss = F.mse_loss(
                noise_pred,
                noise,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)

        print(f"Epoch {epoch}: loss médio = {avg_loss:.6f}")

        if (
            epoch % config["save_every"] == 0
            or epoch == config["epochs"]
        ):

            checkpoint_path = os.path.join(
                config["output_dir"],
                f"diffusion_epoch{epoch}.pt",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )

            print(f"Checkpoint salvo em: {checkpoint_path}")

    final_model_path = os.path.join(
        config["output_dir"],
        "diffusion_final.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        final_model_path,
    )

    print(f"\nModelo final salvo em: {final_model_path}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    train()