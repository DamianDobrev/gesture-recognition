import cv2

from config import CONFIG
from image_processing.image_processor import to_50x50_monochrome

size = CONFIG['size']


def convert_image(ip, img):
    def extract_bounding_boxes_by_skin_threshold(image):
        skin = ip.extract_skin(image)

        binary_skin = ip.hsv_to_binary(skin)
        mask_binary = ip.find_largest_connected_component(binary_skin)

        # Find bounding boxes.
        bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
        square_bbox = ip.get_square_bbox(bbox, image)
        return skin, mask_binary, bbox, square_bbox

    def get_center_hsv(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
        center_px_hsv = img_hsv[int(size / 2), int(size / 2)]
        return center_px_hsv

    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(img)

    # Create image from the original with red/green boxes to show the boundary.
    frame_with_rect_bbox = ip.add_bounding_box_to_img(img, bbox)
    frame_with_rect_sq_bboxes = ip.add_bounding_box_to_img(frame_with_rect_bbox, sq_bbox, (0, 255, 0))

    # Crop frame and binary mask to the correct bounding box.
    hand = ip.crop_image_by_square_bbox(img, sq_bbox, size)
    hand_binary_mask = cv2.cvtColor(ip.crop_image_by_square_bbox(binary_mask, sq_bbox, size), cv2.COLOR_GRAY2RGB)

    return {
        'orig': img,
        'orig_bboxes': frame_with_rect_sq_bboxes,
        'skin': skin,
        'hand': hand,
        'binary_mask': cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB),
        'hand_binary_mask': hand_binary_mask,
        'sq_bbox': sq_bbox,
        'center_hsv': get_center_hsv(img)
    }


def convert_img_for_test_or_prediction(ip, img):
    params = convert_image(ip, img)
    new_img = params['hand_binary_mask']
    new_img = to_50x50_monochrome(new_img)
    return new_img, params
