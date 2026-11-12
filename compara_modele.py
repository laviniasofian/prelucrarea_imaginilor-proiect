import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

ORIGINAL = ROOT / "rezultate" / "pancreas_si_tumora" / "scoruri_validare.csv"
GAUSSIAN = ROOT / "rezultate" / "pancreas_si_tumora_gaussian3x3" / "scoruri_validare.csv"


def citeste(fisier):
    with fisier.open(encoding="utf-8-sig", newline="") as f:
        randuri = list(csv.DictReader(f))
    randuri = [r for r in randuri if 1 <= int(r["epoca"]) <= 20]
    randuri.sort(key=lambda r: int(r["epoca"]))
    if [int(r["epoca"]) for r in randuri] != list(range(1, 21)):
        raise ValueError(f"{fisier.name}: sunt necesare epocile 1-20.")
    return randuri


original = citeste(ORIGINAL)
gaussian = citeste(GAUSSIAN)

metrici = [
    ("dice_tumora", "Dice tumora"),
    ("dice_pancreas", "Dice pancreas"),
    ("loss_train", "Loss antrenare"),
]

fig, axe = plt.subplots(1, 3, figsize=(14, 4))
epoci = list(range(1, 21))

print("COMPARATIE LA EPOCA 20")
for ax, (coloana, titlu) in zip(axe, metrici):
    valori_o = [float(r[coloana]) for r in original]
    valori_g = [float(r[coloana]) for r in gaussian]

    ax.plot(epoci, valori_o, label="Original")
    ax.plot(epoci, valori_g, label="Gaussian 3x3")
    ax.set(title=titlu, xlabel="Epoca")
    ax.grid(alpha=0.3)
    ax.legend()

    print(f"{titlu}: Original={valori_o[-1]:.4f}, Gaussian={valori_g[-1]:.4f}")

fig.suptitle("U-Net 2D: imagini originale vs. Gaussian 3x3")
fig.tight_layout()
fig.savefig(ROOT / "rezultate" / "comparatie_modele" / "comparatie_unet.png", dpi=150)
print("\nGrafic salvat: comparatie_unet.png")
plt.show()
