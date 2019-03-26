import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Prediction parameters.
    # ...
    'predictor_model_dir': '2019-03-25__02-18-07_monochrome-boosted-extracted-skin-99',
    # 'predictor_model_dir': '2019-03-26__01-45-22',
    # The certainty percentage for a gestures in order for it
    # to be considered predicted successfully.
    'predicted_val_threshold': 95.00,

    # Collecting data parameters:
    'class_to_collect_data': 1,

    # Training parameters.
    # ...
    'num_training_samples': 680,
    # This will also be the folder to which we save images after data collection.
    'path_to_raw': os.path.join(package_directory, 'training', 'captured_images'),
    'num_epochs': 10,
    'batch_size': 128,
    'training_img_size': 50,

    # Others.
    # ...
    'path_to_results': os.path.join(package_directory, '__results'),
    'path_to_hsv_ranges_csv': os.path.join(package_directory, 'training', 'hsv_ranges.csv'),
    'classes': ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock'],
    'size': 200,
}