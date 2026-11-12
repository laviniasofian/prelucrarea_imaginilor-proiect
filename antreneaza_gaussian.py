import torch
import torch.nn.functional as F
import antreneaza_unet_pancreas as baza

baza.OUTPUT = baza.ROOT / "rezultate" / "pancreas_si_tumora_gaussian3x3"
baza.BEST_PATH = baza.OUTPUT / "model_best.pth"
baza.LAST_PATH = baza.OUTPUT / "model_ultima_epoca.pth"

citeste_original = baza.citeste_caz


def citeste_cu_gaussian(case_id):
    x, y = citeste_original(case_id)

    kernel = x.new_tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]).reshape(1, 1, 3, 3) / 16

    filtrate = []
    for lot in x.split(baza.BATCH_SIZE):
        lot = F.pad(lot, (1, 1, 1, 1), mode="replicate")
        filtrate.append(F.conv2d(lot, kernel))

    return torch.cat(filtrate), y


if __name__ == "__main__":
    baza.citeste_caz = citeste_cu_gaussian
    baza.main()