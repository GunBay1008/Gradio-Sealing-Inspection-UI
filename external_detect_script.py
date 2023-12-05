import argparse
import os

from PIL import Image
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_yolo(selected_model, image_files, conf_value):
    # Ensure 'temp' directory exists
    # s.makedirs('temp', exist_ok=True)

    # # Save the uploaded images and prepare the file paths
    # image_paths = []
    # for i, img in enumerate(image_files):
    #     path = f"temp/uploaded_image_{i}.jpg"
    #     img = path
    #     image_paths.append(path)

    model = YOLO(selected_model)

    # for image in image_paths:

    results = model.predict(image_files, conf=conf_value, save=True)

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        detected_image = im.save("results.jpg")  # save image
    return detected_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO Training")
    parser.add_argument("selected_model", type=str, help="Selected YOLO model")
    parser.add_argument("image_files", type=str, help="Selecte images")
    parser.add_argument("conf_value", type=float, help="Batch size for training")

    args = parser.parse_args()

    detect_yolo(args.selected_model, args.image_files, args.conf_value)
