# Gradio App
import os
import shutil
import subprocess

import gradio as gr


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

    # temp_dir = "temp"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # print(selected_model)

    # model_file_path = os.path.join(temp_dir, "model.pt")
    # with open(model_file_path, "wb") as file:
    #     file.write(selected_model)
    # print(model_file_path)

    # image_file_path = os.path.join(temp_dir, "pic.jpeg")
    # with open(image_file_path, "wb") as file:
    #     file.write(image_files)
    # print(image_file_path)

    # Use the selected_model path directly, as Gradio provides the file path

    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Assuming selected_model and image_files are file paths
    # Move the selected model file
    model_file_path = shutil.move(
        selected_model, os.path.join(temp_dir, os.path.basename(selected_model))
    )

    # Move the image file
    image_file_path = shutil.move(
        image_files, os.path.join(temp_dir, os.path.basename(image_files))
    )

    subprocess.Popen(
        [
            "python",
            "external_detect_script.py",
            model_file_path,
            image_file_path,
            str(conf_value),
        ]
    )


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

            # When the button is clicked, call start_training
            train_button.click(
                fn=start_training,
                inputs=[model_dropdown, data_file, epochs_count, batch_size],
                outputs=train_status_output,
            )

        with gr.Tab("Detection"):
            # Define the inputs for detection
            selected_model = gr.File(label="Model Path (.pt file)", type="filepath")
            image_files = gr.Image(label="Upload images", type="filepath")
            conf_value = gr.Slider(0.0, 1.0, label="Conf Value")
            detect_button = gr.Button("Start Detection")
            detection_output = gr.Image(label="Detection Output")

            # When the button is clicked, call start_detection
            detect_button.click(
                fn=start_detection,
                inputs=[selected_model, image_files, conf_value],
                outputs=detection_output,
            )

# Launch the app with the tabbed interface
demo.launch(share=True)
