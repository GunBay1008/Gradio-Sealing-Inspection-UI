# Gradio App
import os
import shutil
import subprocess

import gradio as gr
from PIL import Image
from ultralytics import YOLO

from external_detect_script import detect_yolo


async def start_training(selected_model, data_file_path, epochs_count, batch_size):
    if data_file_path is None:
        return "Please upload a data file."
    elif selected_model is None:
        return "Please choose a pre-trained model"

    if not os.path.exists("temp"):
        os.makedirs("temp")

    with open("temp/training_status.txt", "w") as file:
        file.write("Training is about to Start!")

    subprocess.Popen(
        [
            "python",
            "external_train_script.py",
            selected_model,
            data_file_path,
            str(epochs_count),
            str(batch_size),
        ]
    )

    # counter = 0
    # while True:
    #     status = check_training_status()
    #     if status == "Completed":
    #         return "Training completed."
    #     else:
    #         # Alternate the message every second
    #         if counter % 2 == 0:
    #             message = "Training started.."
    #         else:
    #             message = "Training started..."
    #         counter += 1
    #         await asyncio.sleep(1)  # Check every second
    #         with open('temp/training_status.txt', 'w') as file:
    #             file.write(message)
    #         with open('temp/training_status.txt', 'r') as file:
    #             temp = file.read()
    #         return temp
    #         # return message

    status = check_training_status()
    if status == "Completed":
        return "Training completed."
    else:
        return "Training in progress"


def check_training_status():
    try:
        with open("temp/training_status.txt", "r") as file:
            status = file.read()
        return status
    except FileNotFoundError:
        return "Training Status Not Found"


def start_detection(selected_model, image_files, conf_value):
    if selected_model is None:
        return "Please choose a .pt file"

    allowed_extensions = {"jpg", "jpeg", "png"}

    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Move the selected model file
    model_file_path = shutil.move(
        selected_model, os.path.join(temp_dir, os.path.basename(selected_model))
    )

    # Handle multiple image files
    image_file_paths = []
    for image_file in image_files:
        if image_file.lower().endswith(tuple(allowed_extensions)):
            image_file_path = shutil.move(
                image_file, os.path.join(temp_dir, os.path.basename(image_file))
            )
            image_file_paths.append(image_file_path)
        else:
            return f"Unsupported file type: {os.path.basename(image_file)}"

    return detect_yolo(model_file_path, image_file_paths, conf_value)


def clear_directory():
    folder_path = r"temp"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print("file successfully deleted!")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print("directory successfully deleted!")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def clear_directory_wrapper():
    clear_directory()
    print("Temp folder cleared!")


with gr.Blocks() as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Training"):
            # Define the inputs for training
            model_dropdown = gr.Dropdown(
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                label="Select Model",
            )
            data_file = gr.File(label="Data File Path (.yaml file)")
            epochs_count = gr.Slider(1, 500, label="Epochs Count")
            batch_size = gr.Slider(1, 128, label="Batch Size")
            train_button = gr.Button("Start Training")
            train_status_output = gr.Textbox(label="Training Status")
            clear_temp_button = gr.Button("Clear Temp Folder")

            # When the button is clicked, call start_training
            train_button.click(
                fn=start_training,
                inputs=[model_dropdown, data_file, epochs_count, batch_size],
                outputs=train_status_output,
            )
            clear_temp_button.click(fn=clear_directory_wrapper, inputs=[], outputs=[])

        with gr.Tab("Detection"):
            # Define the inputs for detection
            selected_model = gr.File(label="Model Path (.pt file)", type="filepath")
            image_files = gr.Files(label="Upload images", type="filepath")
            conf_value = gr.Slider(0.0, 1.0, label="Conf Value")
            detect_button = gr.Button("Start Detection")
            detection_output = gr.Gallery(label="Detection Output")
            clear_temp_button = gr.Button("Clear Temp Folder")

            # When the button is clicked, call start_detection
            detect_button.click(
                fn=start_detection,
                inputs=[selected_model, image_files, conf_value],
                outputs=detection_output,
            )

            clear_temp_button.click(fn=clear_directory_wrapper, inputs=[], outputs=[])

# Launch the app with the tabbed interface
demo.launch(share=True)
