from pathlib import Path
from zipfile import ZipFile

from ultralytics import YOLO


DATASET_ZIP = Path("crack.v1-wecar_crack_image.yolov8.zip")
DATASET_DIR = Path("crack_dataset")
DATA_YAML = DATASET_DIR / "data.yaml"
BASE_MODEL = "yolov8n.pt"
RUN_NAME = "crack_wall_model"


def prepare_dataset():
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"Dataset zip not found: {DATASET_ZIP}")

    if not DATA_YAML.exists():
        DATASET_DIR.mkdir(exist_ok=True)
        with ZipFile(DATASET_ZIP) as zip_file:
            zip_file.extractall(DATASET_DIR)

    DATA_YAML.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 1",
                "names: ['crack']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return DATA_YAML


def main():
    data_yaml = prepare_dataset()
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name=RUN_NAME,
        exist_ok=True,
    )

    print("Training complete.")
    print(f"Use this model: {Path('runs/detect') / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
