import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.utils.preprocess import preprocessing_num_channels

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from pathlib import Path
from typing import List, Tuple

def find_input_resolutions(data_dir: str, split: str = "train") -> List[Tuple[int,int]]:
    """
    Parcourt data_dir/<split>/images à la recherche d'un .pt exemple,
    le charge (list[Tensor]) et retourne la liste des résolutions (H,W)
    de chaque tenseur dans la liste.
    """
    images_dir = Path(data_dir) / split / "data"
    # on prend juste le premier fichier .pt qu'on trouve
    example_pt = next(images_dir.glob("*.pt"), None)
    if example_pt is None:
        raise FileNotFoundError(f"Aucun .pt trouvé dans {images_dir}")

    specs = torch.load(example_pt, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of Tensors, got {type(specs)} in {example_pt}")

    resolutions = []
    for i, spec in enumerate(specs):
        if not torch.is_tensor(spec):
            raise ValueError(f"Element {i} in {example_pt} is a {type(spec)}, not a Tensor")
        # spec peut être (C,H,W) ou (H,W)
        if spec.ndim == 3:
            _, H, W = spec.shape
        elif spec.ndim == 2:
            H, W = spec.shape
        else:
            raise ValueError(f"Unexpected ndim={spec.ndim} for element {i} in {example_pt}")
        resolutions.append((H, W))

    return resolutions

def main():
    preprocessing = "spectrogram_psnr"
    input_channels = preprocessing_num_channels(preprocessing)
    input_res = find_input_resolutions("/data/RAWSIM/RMA/rf_dataset_thesis")
    # input_res = [(256,256)]
    print('input_res == ', input_res)
    device = 'cuda:0'
    model = MR_YOLO(
        num_classes=20, device=device, reg_max=16,
        output_dir='/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/rf_dataset_thesis/MRS_YOLO_vs_bis',
        input_resolutions=input_res, in_ch=input_channels, width_mult=2/4,
        backbone_mode='TFSep_pyramid', outfusion_channels_mult=1
    )

    # weights_path = '/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/fuse_with_C2PSA/mr_yolovn_outfusion1/last.pt'
    
    # state_dict = torch.load(weights_path, map_location=device)

    # # supprime toutes les clés inutiles
    # clean_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    # missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)

    # ✅ inutile de refaire torch.load ici
    model.fit(
        data_dir="/data/RAWSIM/RMA/rf_dataset_thesis",
        batch_size=64,
        dataset='fused', 
        preprocessing=preprocessing,
        epochs=300, 
        patience=30,
    )

if __name__ == "__main__":
    main()
