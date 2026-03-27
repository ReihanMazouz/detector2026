import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.preprocess import preprocessing_num_channels


def main():
    data_dir = "/data/RAWSIM/RMA/rf_dataset_thesis"
    output_dir = "/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/rf_dataset_thesis/yolov11_complex_specificres"

    preprocessing = "complex_real_imag"
    input_channels = preprocessing_num_channels(preprocessing)

    res_key = "cfg512"
    res_hw = (256, 256)

    device = "cuda:0"
    num_classes = 20
    batch_size = 64
    epochs = 300
    patience = 30
    lr = 1e-3
    width_mult = 2 / 4
    reg_max = 16

    print("preprocessing =", preprocessing)
    print("input_channels =", input_channels)
    print("res_key =", res_key)
    print("res_hw =", res_hw)

    model = YOLOv11(
        output_dir=output_dir,
        num_classes=num_classes,
        reg_max=reg_max,
        device=device,
        input_canals=input_channels,
        width_mult=width_mult,
    )

    model.fit(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        dataset="specificres",
        preprocessing=preprocessing,
        select_res={"res_hw": res_hw, "res_key": res_key},
    )


if __name__ == "__main__":
    main()
