# Gradio App
import os
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


iface = gr.Interface(
    fn=start_training,
    inputs=[
        gr.Dropdown(
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            label="Select Model",
        ),
        gr.File(label="Data File Path (.yaml file)"),
        gr.Slider(1, 500, 100, label="Epochs Count"),
        gr.Slider(1, 128, 32, label="Batch Size"),
    ],
    outputs="text",
)

iface.launch(share=True)
