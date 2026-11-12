
"""U-Net 2D: fundal (0), tumora (1), pancreas (2), pe PANTHER Task1."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F

from pregatire_date import images_dir, incarca_volum, labels_dir
from unet_mic import UNetMic


ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "rezultate" / "impartire_task1.json"
OUTPUT = ROOT / "rezultate" / "pancreas_si_tumora"
BEST_PATH = OUTPUT / "model_best.pth"
LAST_PATH = OUTPUT / "model_ultima_epoca.pth"
SEED = 42
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def citeste_caz(case_id):
    x, y = incarca_volum(
        images_dir / f"{case_id}_0000.mha", labels_dir / f"{case_id}.mha"
    )
    if not (y == 1).any():
        raise ValueError(f"{case_id}: eticheta tumorii lipseste dupa pregatire.")
    return x.float(), y.long()


def sectiuni_antrenare(y, rng):
    cu_tumora = (y == 1).any(dim=(1, 2))
    cu_pancreas = (y == 2).any(dim=(1, 2))
    grupuri = (
        (cu_tumora, 6),
        (cu_pancreas & ~cu_tumora, 3),
        (~cu_tumora & ~cu_pancreas, 3),
    )
    indici = []
    for grup, limita in grupuri:
        candidati = torch.where(grup)[0].tolist()
        indici.extend(rng.sample(candidati, min(limita, len(candidati))))
    rng.shuffle(indici)
    return indici


def pierdere(logits, y, ponderi):
    ce = F.cross_entropy(logits, y, weight=ponderi)
    p = logits.softmax(dim=1)[:, 1:]
    t = F.one_hot(y, num_classes=3).permute(0, 3, 1, 2)[:, 1:].float()
    dimensiuni = (0, 2, 3)
    intersectie = (p * t).sum(dim=dimensiuni)
    total = p.sum(dim=dimensiuni) + t.sum(dim=dimensiuni)
    dice = (2 * intersectie + 1e-6) / (total + 1e-6)
    return ce + 1 - dice.mean()


def antreneaza(model, optimizer, cazuri, rng):
    model.train()
    cazuri = list(cazuri)
    rng.shuffle(cazuri)
    ponderi = torch.tensor([1.0, 12.0, 4.0], device=DEVICE)
    total_loss = 0.0
    total_sectiuni = 0

    for index, case_id in enumerate(cazuri, start=1):
        x, y = citeste_caz(case_id)
        indici = sectiuni_antrenare(y, rng)

        for start in range(0, len(indici), BATCH_SIZE):
            lot = indici[start:start + BATCH_SIZE]
            imagini, masti = x[lot].to(DEVICE), y[lot].to(DEVICE)
            if rng.random() < 0.5:
                imagini = imagini.flip(dims=(-1,))
                masti = masti.flip(dims=(-1,))

            optimizer.zero_grad(set_to_none=True)
            loss = pierdere(model(imagini), masti, ponderi)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lot)
            total_sectiuni += len(lot)

        if index == 1 or index % 8 == 0 or index == len(cazuri):
            print(f"  Antrenare {index}/{len(cazuri)} volume", flush=True)

    return total_loss / total_sectiuni


@torch.no_grad()
def valideaza(model, cazuri):
    model.eval()
    scoruri = {1: [], 2: []}
    exemple = []
    volume_cu_predictie = 0

    for index, case_id in enumerate(cazuri, start=1):
        x, y = citeste_caz(case_id)
        z = int((y == 1).sum(dim=(1, 2)).argmax())
        intersectii = torch.zeros(2, dtype=torch.int64, device=DEVICE)
        totaluri = torch.zeros_like(intersectii)
        predictie_exemplu = None
        are_predictie_tumora = torch.zeros((), dtype=torch.bool, device=DEVICE)

        for start in range(0, len(x), BATCH_SIZE):
            stop = start + BATCH_SIZE
            predictie = model(x[start:stop].to(DEVICE)).argmax(dim=1)
            referinta = y[start:stop].to(DEVICE)
            are_predictie_tumora |= (predictie == 1).any()

            for pozitie, clasa in enumerate((1, 2)):
                pred = predictie == clasa
                real = referinta == clasa
                intersectii[pozitie] += (pred & real).sum()
                totaluri[pozitie] += pred.sum() + real.sum()

            if index <= 2 and start <= z < min(stop, len(x)):
                predictie_exemplu = predictie[z - start].cpu()

        volume_cu_predictie += int(are_predictie_tumora.item())
        for clasa, comune, total in zip(
            (1, 2), intersectii.tolist(), totaluri.tolist()
        ):
            if total:
                scoruri[clasa].append(2 * comune / total)

        if predictie_exemplu is not None:
            exemple.append((case_id, z, x[z, 0].clone(), y[z].clone(), predictie_exemplu))

        if index % 4 == 0 or index == len(cazuri):
            print(f"  Validare {index}/{len(cazuri)} volume", flush=True)

    medii = {
        clasa: sum(valori) / len(valori) if valori else 0.0
        for clasa, valori in scoruri.items()
    }
    return medii, exemple, volume_cu_predictie


def salveaza_figura(exemple, epoch):
    cmap = ListedColormap(["black", "#ff4d6d", "#ffda55"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(len(exemple), 3, figsize=(11, 4 * len(exemple)), squeeze=False)

    for rand, (case_id, z, imagine, reala, prezisa) in zip(axes, exemple):
        titluri = (f"RMN {case_id} | sectiunea {z}", "Masca de referinta", "Predictie U-Net")

        for ax, titlu, masca in zip(rand, titluri, (None, reala, prezisa)):
            ax.imshow(imagine.numpy(), cmap="gray")
            if masca is not None:
                ax.imshow(
                    masca.numpy(), cmap=cmap, norm=norm,
                    alpha=(masca != 0).numpy() * 0.65,
                    interpolation="nearest",
                )
            ax.set_title(titlu)
            ax.axis("off")

    fig.suptitle(f"Validare | modelul din epoca {epoch}")
    fig.legend(
        handles=[Patch(color="#ff4d6d", label="Tumora"), Patch(color="#ffda55", label="Pancreas")],
        loc="lower center", ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    temporar = OUTPUT / "predictii_validare.tmp"
    fig.savefig(temporar, format="png", dpi=150)
    plt.close(fig)
    temporar.replace(OUTPUT / "predictii_validare.png")


def salveaza_model(checkpoint, path):
    temporar = path.with_suffix(".tmp")
    torch.save(checkpoint, temporar)
    temporar.replace(path)


def salveaza_istoric(istoric):
    with (OUTPUT / "scoruri_validare.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(istoric[0]))
        writer.writeheader()
        writer.writerows(istoric)


def main():
    parser = argparse.ArgumentParser(description="Antrenare U-Net pentru pancreas si tumora")
    parser.add_argument("--epoci", type=int, default=20, help="Numarul total de epoci")
    args = parser.parse_args()
    if args.epoci < 1:
        parser.error("--epoci trebuie sa fie cel putin 1")

    torch.manual_seed(SEED)
    with SPLIT_PATH.open(encoding="utf-8") as file:
        split = json.load(file)
    if not split["train"] or not split["val"]:
        raise ValueError("Impartirea trebuie sa contina cazuri pentru train si val.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = UNetMic(n_channels=1, n_classes=3).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    prima_epoca = 1
    scor_optim = -1.0
    istoric = []

    if LAST_PATH.is_file():
        ultimul = torch.load(LAST_PATH, map_location=DEVICE, weights_only=True)
        if ultimul["split"] != split:
            raise ValueError("Impartirea s-a schimbat fata de modelul salvat.")
        model.load_state_dict(ultimul["model_state_dict"])
        optimizer.load_state_dict(ultimul["optimizer_state_dict"])
        prima_epoca = ultimul["epoch"] + 1
        scor_optim = ultimul["best_score"]
        istoric = ultimul["history"]

    print(
        f"Dispozitiv: {DEVICE} | Train: {len(split['train'])} | Val: {len(split['val'])}",
        flush=True,
    )

    for epoch in range(prima_epoca, args.epoci + 1):
        start = time.perf_counter()
        print(f"\nEpoca {epoch}/{args.epoci}", flush=True)
        loss = antreneaza(model, optimizer, split["train"], random.Random(SEED + epoch))
        dice, exemple, volume_marcate = valideaza(model, split["val"])

        # Tumora are prioritate, dar urmarim si pancreasul.
        scor = 0.7 * dice[1] + 0.3 * dice[2]
        istoric.append({
            "epoca": epoch, "loss_train": loss, "dice_tumora": dice[1],
            "dice_pancreas": dice[2], "volume_cu_predictie_tumora": volume_marcate,
        })
        print(
            f"Loss {loss:.4f} | Dice tumora {dice[1]:.4f} | Dice pancreas {dice[2]:.4f} | "
            f"Tumora prezisa in {volume_marcate}/{len(split['val'])} volume | "
            f"{(time.perf_counter() - start) / 60:.1f} min",
            flush=True,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "dice_tumora": dice[1],
            "dice_pancreas": dice[2],
            "n_channels": 1,
            "n_classes": 3,
        }
        if scor >= scor_optim:
            scor_optim = scor
            salveaza_model(checkpoint, BEST_PATH)
            salveaza_figura(exemple, epoch)
            print("  Modelul si imaginea de validare au fost actualizate.", flush=True)

        salveaza_model({
            **checkpoint,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": scor_optim,
            "history": istoric,
            "split": split,
        }, LAST_PATH)
        salveaza_istoric(istoric)

    print(f"\nRezultate: {OUTPUT}", flush=True)
    print("Deschide predictii_validare.png pentru comparatia vizuala.", flush=True)


if __name__ == "__main__":
    main()



