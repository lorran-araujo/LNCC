from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """Carrega todas as imagens de uma pasta (recursivamente) e aplica transforms."""

    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, root_dir, image_size=64):
        self.paths = [
            p for p in Path(root_dir).rglob("*")
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise ValueError(f"Nenhuma imagem encontrada em: {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                 # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)