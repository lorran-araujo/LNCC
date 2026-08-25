import argparse
import os

import torch
import yaml
from torchvision.utils import make_grid, save_image

from ddpm import NoiseScheduler
from model import SimpleUNet
from sampling import sample_ddpm


def parse_args():
    parser = argparse.ArgumentParser(description="Gera uma prévia de amostras de um checkpoint DDPM.")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo de configuração.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint a carregar. Se omitido, usa config['checkpoint'].")
    parser.add_argument("--output-dir", default=None, help="Pasta para salvar as imagens. Se omitida, usa config['sample_output_dir'].")
    parser.add_argument("--num-images", type=int, default=None, help="Número de imagens a gerar. Se omitido, usa config['num_images'].")
    parser.add_argument(
        "--save-intermediates",
        type=int,
        default=None,
        help="Intervalo para salvar passos de denoising. Se omitido, usa config['save_intermediates'].",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed opcional para reproduzir a geração.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checkpoint_path = args.checkpoint or config["checkpoint"]
    output_dir = args.output_dir or config["sample_output_dir"]
    num_images = args.num_images or config.get("num_images", 8)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. Rode python3 train.py primeiro ou passe --checkpoint."
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint["config"]

    model = SimpleUNet(
        in_channels=3,
        base_ch=train_config["base_channels"],
    ).to(device)
    state_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    print(f"Pesos carregados: {state_key}")

    scheduler = NoiseScheduler(
        num_timesteps=train_config["timesteps"],
        device=device,
    )

    save_intermediates = args.save_intermediates
    if save_intermediates is None:
        save_intermediates = config.get("save_intermediates", 0)

    images, intermediates = sample_ddpm(
        model=model,
        scheduler=scheduler,
        num_images=num_images,
        image_size=train_config["image_size"],
        device=device,
        save_intermediates=save_intermediates,
    )
    images = (images + 1) / 2

    os.makedirs(output_dir, exist_ok=True)

    for index, image in enumerate(images):
        save_image(image, os.path.join(output_dir, f"preview_{index:03d}.png"))

    grid = make_grid(images, nrow=min(num_images, 4))
    grid_path = os.path.join(output_dir, "preview_grid.png")
    save_image(grid, grid_path)

    if save_intermediates > 0 and intermediates:
        steps_dir = os.path.join(output_dir, "denoising_steps")
        os.makedirs(steps_dir, exist_ok=True)

        for index, step_batch in enumerate(intermediates):
            step_img = (step_batch[0] + 1) / 2
            save_image(step_img, os.path.join(steps_dir, f"step_{index:03d}.png"))

        print(f"Passos intermediários salvos em: {steps_dir}")

    print(f"Imagens salvas em: {output_dir}")
    print(f"Grid salvo em: {grid_path}")


if __name__ == "__main__":
    main()
