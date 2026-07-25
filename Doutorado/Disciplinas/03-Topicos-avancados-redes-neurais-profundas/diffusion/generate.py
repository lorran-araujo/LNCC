"""
Geração de imagens (sampling) a partir de um modelo de difusão treinado.

Os parâmetros são lidos do arquivo config.yaml.
"""

import os
import yaml

import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model import SimpleUNet
from ddpm import NoiseScheduler


# -------------------------------------------------
# Carrega configuração
# -------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuração carregada:")
print(config)


@torch.no_grad()
def sample(
    model,
    scheduler,
    num_images,
    image_size,
    device,
    save_intermediates=None,
):
    """
    Executa o processo reverso do DDPM.
    """

    model.eval()

    T = scheduler.num_timesteps

    x = torch.randn(
        num_images,
        3,
        image_size,
        image_size,
        device=device,
    )

    betas = scheduler.betas
    alphas = scheduler.alphas
    alphas_cumprod = scheduler.alphas_cumprod

    alphas_cumprod_prev = torch.cat(
        [
            torch.tensor([1.0], device=device),
            alphas_cumprod[:-1],
        ]
    )

    intermediates = []

    for t in tqdm(
        reversed(range(T)),
        total=T,
        desc="Sampling",
    ):

        t_batch = torch.full(
            (num_images,),
            t,
            device=device,
            dtype=torch.long,
        )

        noise_pred = model(x, t_batch)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - coef * noise_pred
        )

        if t > 0:

            alpha_cumprod_prev_t = alphas_cumprod_prev[t]

            posterior_var = (
                beta_t
                * (1 - alpha_cumprod_prev_t)
                / (1 - alpha_cumprod_t)
            )

            noise = torch.randn_like(x)

            x = mean + torch.sqrt(posterior_var) * noise

        else:

            x = mean

        if (
            save_intermediates is not None
            and save_intermediates > 0
            and t % save_intermediates == 0
        ):
            intermediates.append(
                x.clamp(-1, 1).cpu()
            )

    return x.clamp(-1, 1), intermediates


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Usando device: {device}")

    checkpoint = torch.load(
        config["checkpoint"],
        map_location=device,
    )

    train_config = checkpoint["config"]

    image_size = train_config["image_size"]
    base_channels = train_config["base_channels"]
    timesteps = train_config["timesteps"]

    print(
        f"Config do modelo -> "
        f"image_size={image_size}, "
        f"base_channels={base_channels}, "
        f"timesteps={timesteps}"
    )

    model = SimpleUNet(
        in_channels=3,
        base_ch=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    scheduler = NoiseScheduler(
        num_timesteps=timesteps,
        device=device,
    )

    os.makedirs(
        config["sample_output_dir"],
        exist_ok=True,
    )

    images, intermediates = sample(
        model=model,
        scheduler=scheduler,
        num_images=config["num_images"],
        image_size=image_size,
        device=device,
        save_intermediates=config["save_intermediates"],
    )

    # [-1,1] -> [0,1]
    images_01 = (images + 1) / 2

    # Salva imagens individuais
    for i in range(images_01.shape[0]):

        save_image(
            images_01[i],
            os.path.join(
                config["sample_output_dir"],
                f"sample_{i:03d}.png",
            ),
        )

    # Salva grid
    grid = make_grid(
        images_01,
        nrow=min(config["num_images"], 4),
    )

    save_image(
        grid,
        os.path.join(
            config["sample_output_dir"],
            "grid.png",
        ),
    )

    print(
        f"{config['num_images']} imagens salvas em "
        f"{config['sample_output_dir']}"
    )

    # Salva passos intermediários
    if (
        config["save_intermediates"] > 0
        and len(intermediates) > 0
    ):

        steps_dir = os.path.join(
            config["sample_output_dir"],
            "denoising_steps",
        )

        os.makedirs(
            steps_dir,
            exist_ok=True,
        )

        for idx, step_batch in enumerate(intermediates):

            step_img = (step_batch[0] + 1) / 2

            save_image(
                step_img,
                os.path.join(
                    steps_dir,
                    f"step_{idx:03d}.png",
                ),
            )

        print(
            f"Passos intermediários salvos em: {steps_dir}"
        )


if __name__ == "__main__":
    main()