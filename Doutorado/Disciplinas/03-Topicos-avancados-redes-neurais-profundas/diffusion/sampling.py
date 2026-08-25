import torch
from tqdm import tqdm


@torch.no_grad()
def sample_ddpm(
    model,
    scheduler,
    num_images,
    image_size,
    device,
    save_intermediates=None,
):
    """Executa o processo reverso do DDPM."""
    model.eval()

    x = torch.randn(num_images, 3, image_size, image_size, device=device)

    alphas_cumprod_prev = torch.cat(
        [
            torch.tensor([1.0], device=device),
            scheduler.alphas_cumprod[:-1],
        ]
    )

    intermediates = []

    for t in tqdm(
        reversed(range(scheduler.num_timesteps)),
        total=scheduler.num_timesteps,
        desc="Sampling",
    ):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)

        beta_t = scheduler.betas[t]
        alpha_t = scheduler.alphas[t]
        alpha_cumprod_t = scheduler.alphas_cumprod[t]

        mean = (x - beta_t * noise_pred / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)

        if t > 0:
            alpha_cumprod_prev_t = alphas_cumprod_prev[t]
            posterior_var = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean

        if save_intermediates is not None and save_intermediates > 0 and t % save_intermediates == 0:
            intermediates.append(x.clamp(-1, 1).cpu())

    return x.clamp(-1, 1), intermediates
