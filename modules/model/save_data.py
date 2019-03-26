import os
import random

import cv2
from matplotlib import pyplot as plt

from config import CONFIG
from modules.common import create_path_if_does_not_exist

img_rows, img_cols = CONFIG['training_img_size'], CONFIG['training_img_size']
batch_size = CONFIG['batch_size']
num_epochs = CONFIG['num_epochs']
results_path = CONFIG['path_to_results']


def save_hist(hist, path_to_save):
    # visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(len(train_loss))

    # Loss.
    fig_loss = plt.figure(1,figsize=(7,5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    # Acc.
    fig_acc = plt.figure(2,figsize=(7,5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()
    fig_acc.savefig(os.path.join(path_to_save, 'accuracy.png'))
    fig_loss.savefig(os.path.join(path_to_save, 'loss.png'))


def save_data_info(file_dir, x_train, x_test, y_train, y_test):
    folder_name = 'test_set_processed'
    data_path = os.path.join(file_dir, folder_name)
    create_path_if_does_not_exist(data_path)

    f = open(os.path.join(file_dir, folder_name, 'data_info.txt'), 'w+')
    f.write('len(x_train): ' + str(len(x_train)) + '\n')
    f.write('len(x_test): ' + str(len(x_test)) + '\n')
    f.close()

    for idx, img in enumerate(x_test):
        img_path = os.path.join(data_path, str(CONFIG['classes'][y_test[idx]]))
        create_path_if_does_not_exist(img_path)
        cv2.imwrite(os.path.join(img_path, 'img-' + str(random.randrange(999999)) + '.png'), img)


def save_notes(file_dir):
    f = open(os.path.join(file_dir, 'notes.txt'), 'w+')
    f.write('# Add notes here about this training/model...' + '\n')
    f.close()


def save_info(file_dir, model, eval):
    f = open(os.path.join(file_dir, 'eval.txt'), 'w+')
    f.write('eval_loss=' + str(eval[0]) + '\n')
    f.write('eval_acc =' + str(eval[1]) + '\n\n')
    f.close()

    f = open(os.path.join(file_dir, 'model_summary.txt'), 'w+')
    f.write(str(model.summary(print_fn=lambda x: f.write(x + '\n'))))
    f.close()

    f = open(os.path.join(file_dir, 'training_summary.txt'), 'w+')
    f.write('img_w,img_h=' + str(img_cols) + ',' + str(img_rows) + '\n')
    f.write('batch_size=' + str(batch_size) + '\n')
    f.write('num_epochs=' + str(num_epochs) + '\n')
    f.write('classes=' + str(CONFIG['classes']))
    f.close()


def save_config(file_dir):
    f = open(os.path.join(file_dir, 'config.csv'), 'w+')
    props = ['training_img_size', 'training_set_name', 'training_set_image_type']
    f.write(','.join(props) + '\n')
    f.write(','.join(map(lambda prop: str(CONFIG[prop]), props)) + '\n')
    f.close()


def save_confusion_matrix(file_dir):
    # TODO implement
    pass
