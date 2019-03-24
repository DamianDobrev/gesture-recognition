from datetime import datetime

import cv2
import imutils

# from sklearn.utils import shuffle
import numpy as np

import os

from training.predictor import predict
from image_processing import image_processor

size = 200
path_to_captured_images = './captured_images/'
path_to_captured_masks = './captured_masks/'

def loop(class_number, milliseconds=200):
    cap = cv2.VideoCapture(0)

    count = 1
    is_saving_images = False
    is_predicting = True
    last_time = datetime.now()

    path_output_dir = path_to_captured_images + str(class_number) + '/'
    path_masks_output_dir = path_to_captured_masks + str(class_number) + '/'

    # ip = image_processor.ImageProcessor(size, [102, 40, 34], [179, 255, 255])  # Works well at home in daylight.
    ip = image_processor.ImageProcessor(size, [104, 25, 34], [179, 255, 180])

    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    if not os.path.exists(path_masks_output_dir):
        os.makedirs(path_masks_output_dir)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame)

        processed = ip.extract_skin(frame)

        thr = ip.hsv_to_binary(processed)
        mask_binary = ip.find_largest_connected_component(thr)

        # Find bounding boxes.
        bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
        frame_with_bbox = ip.add_bounding_box_to_img(frame, bbox)
        square_bbox = ip.get_square_bbox(bbox, frame)
        frame_with_bboxes = ip.add_bounding_box_to_img(frame_with_bbox, square_bbox, (0, 255, 0))

        # Crop frame to the correct bounding box.
        cropped_image = ip.crop_image_by_square_bbox(frame, square_bbox, size)

        # Also crop the binary mask.
        cropped_binary_mask = ip.crop_image_by_square_bbox(mask_binary, square_bbox, size)
        cropped_binary_mask = cv2.cvtColor(cropped_binary_mask, cv2.COLOR_GRAY2BGR)

        prediction_value = np.zeros((200, 200, 3), np.uint8)
        # Manage input.
        if is_predicting:
            class_num, values, text = predict(cropped_binary_mask)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (50, 50)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            cv2.putText(prediction_value, text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        elif is_saving_images:
            cur_time = datetime.now()
            time_diff = cur_time - last_time
            time_diff_milliseconds = time_diff.total_seconds() * 1000
            if time_diff_milliseconds >= milliseconds:
                cv2.imwrite(os.path.join(path_output_dir, 'raw_%03d.png') % count, cropped_image)
                cv2.imwrite(os.path.join(path_masks_output_dir, 'raw_%02d.png') % count, cropped_binary_mask)
                count += 1
                print(count)
        else:
            print('S -> start saving images of class [' + str(class_number) + ']')
            print('P -> predict...')

        # Visualize.
        window_name = 'Img with Bbox + processing.'
        height, width = frame_with_bboxes.shape[:2]
        cv2.rectangle(frame_with_bboxes, (int(width/2-3), int(height/2-3)), (int(width/2 + 3), int(height/2+3)), (255, 255, 0), 1)
        cv2.imshow(window_name, np.hstack([frame_with_bboxes, processed, cropped_image, cropped_binary_mask, prediction_value]))

        # Handle input.
        if cv2.waitKey(1):
            if cv2.waitKey(1) & 0xFF == ord('s'):
                is_saving_images = not is_saving_images
            if cv2.waitKey(1) & 0xFF == ord('b'):
                is_predicting = not is_predicting
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

loop(999)
cv2.waitKey(0)