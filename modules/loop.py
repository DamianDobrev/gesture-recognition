import cv2
import imutils

from config import CONFIG

size = CONFIG['size']


def loop(fn, ip):
    cap = cv2.VideoCapture(0)
    cap.set(15, 0.00001)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame, size)
        should_break = fn(ip, frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()


# def add_folder():
#     if not os.path.exists(path_to_captured_masks):
#         os.makedirs(path_to_captured_masks)
#
#     imgs = fetch_training_images(path_to_captured_images, 700)
#     for i in range(0,7):
#         img_list = imgs[i][0]
#         class_num = imgs[i][1]
#         print(np.array(img_list).shape)
#         print(class_num)
#
#         for j, img in enumerate(img_list):
#             # img = np.moveaxis(img, -1, 0)
#             # img = np.moveaxis(img, -1, 0)
#             # print('dasdsa', img.shape)
#             # cv2.imshow('shittty', img)
#             # cv2.waitKey(0)
#             # print('kur', path_to_captured_masks + str(class_num) + '/')
#             def save_masked_img(img, frame_with_rect_sq_bboxes, skin, hand, binary_mask, hand_binary_mask, sq_bbox):
#                 print('kur', path_to_captured_masks + str(class_num + 1) + '/')
#                 path = path_to_captured_masks + str(class_num + 1) + '/'
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#
#                 cv2.imwrite(os.path.join(path, 'binary_%03d.png') % (j+1), hand_binary_mask)
#             run(img, save_masked_img)

# add_folder()