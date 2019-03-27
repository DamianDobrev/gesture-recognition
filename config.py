import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Prediction parameters.
    # ...
    'predictor_model_dir': '2019-03-26__22-34-50_num_img_1200',
    # The certainty percentage for a gestures in order for it
    # to be considered predicted successfully.
    'predicted_val_threshold': 98.00,

    # Collecting data parameters:
    'data_collect_class': 10,
    'data_collect_set_name': 'test',  # !!! Contents here will be overwritten by the DataCollector.

    # Training parameters.
    # ...
    'num_training_samples': 2000,
    'training_set_name': 'min_1200_per_class',
    'training_set_image_type': 'skin_monochrome',
    'num_epochs': 15,
    'batch_size': 128,
    'training_img_size': 100,
    'test_split': 0.3,
    'validation_split': 0.2,

    # General.
    # ...
    'training_data_path': os.path.join(package_directory, '__training_data'),
    'results_path': os.path.join(package_directory, '__results'),
    'hsv_ranges_path': os.path.join(package_directory, 'hsv_ranges.csv'),
    'classes': ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock'],
    'size': 200,
    'imshow_window_name': 'Gesture Recognition',
    'bbox_threshold': 20
}