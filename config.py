import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Prediction parameters.
    # ...
    'predictor_model_dir': '2019-03-26__22-34-50_num_img_1200',
    # 'predictor_model_dir': '2019-03-26__01-45-22',
    # The certainty percentage for a gestures in order for it
    # to be considered predicted successfully.
    'predicted_val_threshold': 98.00,

    # Collecting data parameters:
    'class_to_collect_data': 10,

    # Training parameters.
    # ...
    'num_training_samples': 400,
    # This will also be the folder to which we save images after data collection.
    # path_to_raw ->
    'training_sets_path': os.path.join(package_directory, '__training_data'),
    'training_set_name': 'min_1200_per_class',
    'training_set_image_type': 'skin_monochrome',
    'num_epochs': 8,
    'batch_size': 16,
    'training_img_size': 50,
    'test_split': 0.4,
    'validation_split': 0.2,

    # Others.
    # ...
    'path_to_results': os.path.join(package_directory, '__results'),
    'path_to_hsv_ranges_csv': os.path.join(package_directory, 'hsv_ranges.csv'),
    'classes': ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock'],
    'size': 200,
    'imshow_window_name': 'Gesture Recognition',
    'bbox_threshold': 20
}