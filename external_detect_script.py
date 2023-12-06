import os

from PIL import Image
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_yolo(selected_model, image_file_paths, conf_value):
    model = YOLO(selected_model)

    detected_images = []
    for image_file in image_file_paths:
        results = model.predict(image_file, conf=conf_value, save=True)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            detected_images.append(im)

    return detected_images
