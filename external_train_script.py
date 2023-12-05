import argparse
import os
import shutil

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_yolo(selected_model, data_file_path, epochs_count, batch_size):
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    data_temp_path = os.path.join(temp_folder, "data.yaml")
    shutil.copy(data_file_path, data_temp_path)

    model = YOLO(selected_model)
    with open("temp/training_status.txt", "w") as file:
        file.write("Training Started")
    results = model.train(data=data_temp_path, epochs=epochs_count, batch=batch_size)
    # At the end of your train_yolo function
    with open("temp/training_status.txt", "w") as file:
        file.write("Completed")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO Training")
    parser.add_argument("selected_model", type=str, help="Selected YOLO model")
    parser.add_argument("data_file_path", type=str, help="Path to data file")
    parser.add_argument("epochs_count", type=int, help="Number of epochs")
    parser.add_argument("batch_size", type=int, help="Batch size for training")

    args = parser.parse_args()

    train_yolo(
        args.selected_model, args.data_file_path, args.epochs_count, args.batch_size
    )
