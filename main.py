import sys
from datetime import datetime
import PyQt5

import cv2
import imutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from image_processing import image_processor
from image_processing.canvas import Canvas
from training.predictor import predict

matplotlib.use('Qt5Agg')

print('Starting...')

size = 200
cap = cv2.VideoCapture(0)
last_time = datetime.now()

window_name = 'Img with Bbox + processing.'

path_to_captured_images = './captured_images/'
path_to_captured_masks = './captured_masks/'
ip = image_processor.ImageProcessor(size, [104, 25, 34], [179, 255, 180])

# is_saving_images = False
# is_predicting = True

def loop():
    # path_output_dir = path_to_captured_images + str(class_number) + '/'
    # path_masks_output_dir = path_to_captured_masks + str(class_number) + '/'

    # if not os.path.exists(path_output_dir):
    #     os.makedirs(path_output_dir)
    # if not os.path.exists(path_masks_output_dir):
    #     os.makedirs(path_masks_output_dir)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame)
        should_break = run(frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()


def find_skin_bbox(image):
    skin = ip.extract_skin(image)

    binary_skin = ip.hsv_to_binary(skin)
    mask_binary = ip.find_largest_connected_component(binary_skin)

    # Find bounding boxes.
    bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
    square_bbox = ip.get_square_bbox(bbox, image)
    return skin, mask_binary, bbox, square_bbox


def append_rectangle_in_center(img, color=(255, 255, 0)):
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect


def create_canvas(size):
    return np.zeros(size, np.uint8)


def visualise(params, orig, skin, binary_mask, hand, hand_binary_mask):
    orig_mark_center = append_rectangle_in_center(orig)

    vals_canvas = Canvas((size, 500, 3))
    vals_canvas.draw_text(1, 'hsv: ' + str(params['center_hsv']))
    vals_canvas.draw_text(2, 'res: ' + str(params['result'][2]))

    empty_canvas = Canvas((size, 500, 3))

    stack1 = np.hstack([orig_mark_center, skin, vals_canvas.print()])
    stack2 = np.hstack([hand, hand_binary_mask, empty_canvas.print()])

    cv2.imshow(window_name, np.vstack([stack1, stack2]))


def run_selected_action(hand, hand_binary_mask):
    # TODO fill in this to be depending on user input.
    # return save_img(hand, hand_binary_mask, 1)  # Class == 1
    return predict(hand_binary_mask)


def get_center_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
    center_px_hsv = img_hsv[int(size / 2), int(size / 2)]
    return center_px_hsv


def run(frame):
    skin, binary_mask, bbox, sq_bbox = find_skin_bbox(frame)

    frame_with_rect_bbox = ip.add_bounding_box_to_img(frame, bbox)
    frame_with_rect_sq_bboxes = ip.add_bounding_box_to_img(frame_with_rect_bbox, sq_bbox, (0, 255, 0))

    # Crop frame and binary mask to the correct bounding box.
    hand = ip.crop_image_by_square_bbox(frame, sq_bbox, size)
    hand_binary_mask = cv2.cvtColor(ip.crop_image_by_square_bbox(binary_mask, sq_bbox, size), cv2.COLOR_GRAY2RGB)

    # Visualise and run user action.
    result = run_selected_action(hand, hand_binary_mask)

    params = {
        'center_hsv': get_center_hsv(frame),
        'result': result
    }

    visualise(params, frame_with_rect_sq_bboxes, skin, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB), hand, hand_binary_mask)

    # Handle input.
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # is_saving_images = not is_saving_images
        print('pressed s')
    if cv2.waitKey(1) & 0xFF == ord('b'):
        # is_predicting = not is_predicting
        print('pressed b')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True


loop()
cv2.waitKey(0)