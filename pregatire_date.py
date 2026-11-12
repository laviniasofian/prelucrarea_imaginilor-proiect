from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
task_dir = root / "data" / "Task1"
images_dir = task_dir / "ImagesTr"
labels_dir = task_dir / "LabelsTr"


def incarca_volum(image_path, label_path):
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))

    if (
        image.GetSize() != label.GetSize()
        or not np.allclose(image.GetSpacing(), label.GetSpacing())
        or not np.allclose(image.GetOrigin(), label.GetOrigin())
        or not np.allclose(image.GetDirection(), label.GetDirection())
    ):
        raise ValueError("Imaginea si masca nu au aceeasi geometrie.")

    x = sitk.GetArrayFromImage(image).astype(np.float32)
    y = sitk.GetArrayFromImage(label).astype(np.int64)

    if not np.isfinite(x).all():
        raise ValueError("Imaginea contine valori invalide.")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError(f"Etichete neasteptate: {np.unique(y)}")

    low, high = np.percentile(x, [1, 99])
    x = np.clip(x, low, high)
    x = (x - x.mean()) / max(float(x.std()), 1e-8)

    x = torch.from_numpy(x).unsqueeze(1)
    y = torch.from_numpy(y).unsqueeze(1)

    x = F.interpolate(
        x, size=(256, 256),
        mode="bilinear", align_corners=False
    )

    y = F.interpolate(
        y.float(), size=(256, 256), mode="nearest"
    ).squeeze(1).long()

    return x, y


if __name__ == "__main__":
    images = sorted(images_dir.glob("*_0000.mha"))

    if not images:
        raise FileNotFoundError(
            f"Nu am gasit imagini in: {images_dir}"
        )

    pairs = []
    for image_path in images:
        case_id = image_path.stem.removesuffix("_0000")
        label_path = labels_dir / f"{case_id}.mha"

        if not label_path.is_file():
            raise FileNotFoundError(f"Lipseste masca: {label_path}")

        pairs.append((image_path, label_path))

    print("Perechi imagine-masca:", len(pairs))

    image_path, label_path = pairs[0]
    x, y = incarca_volum(image_path, label_path)

    print("Imagine:", image_path.name)
    print("Forma imaginilor:", tuple(x.shape))
    print("Forma mastilor:", tuple(y.shape))
    print("Clase dupa pregatire:", torch.unique(y).tolist())

    z = int((y != 0).sum(dim=(1, 2)).argmax())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(x[z, 0].numpy(), cmap="gray")
    axes[0].set_title("RMN pregatit pentru U-Net")

    axes[1].imshow(
        y[z].numpy(), cmap="viridis",
        vmin=0, vmax=2, interpolation="nearest"
    )
    axes[1].set_title("Masca: valorile 0, 1, 2")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()