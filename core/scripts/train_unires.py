import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov8 import YOLOv8
from detector2026.core.utils.preprocess import preprocessing_num_channels

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    preprocessing = "spectrogram_psnr"
    input_channels = preprocessing_num_channels(preprocessing)
    device = 'cuda:0'
    model = YOLOv8(
        num_classes=20, device=device, reg_max=16,
        output_dir='/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/rf_dataset_thesis/yolov8s_specificres',
        in_ch=input_channels,
        width_mult=2/4,
    )

    # weights_path = '/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/iclr_rf_simulator/yolov11_cfg2048/last.pt'
    # model.load_weights(weights_path)

    # ✅ inutile de refaire torch.load ici
    model.fit(
        data_dir="/data/RAWSIM/RMA/rf_dataset_thesis",
        batch_size=64,
        dataset='specificres', 
        preprocessing=preprocessing,
        epochs=300, 
        patience=30,
        select_res={"res_hw":(256,256), 'res_key':'cfg512'}
    )

if __name__ == "__main__":
    main()
