import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'path_to_results': os.path.join(package_directory, '__results'),
    'path_to_hsv_ranges_csv': os.path.join(package_directory, 'training', 'hsv_ranges.csv'),
    'path_to_raw': os.path.join(package_directory, 'training', 'captured_images'),
    'size': 200,
    'predictor_model_dir': '2019-03-25__02-18-07',
    # Since the batch size is 32, and 80% of data is training, 680*0.8 % 32 should equal 0!
    # This is why 680 instead of 700:
    'num_training_samples': 680,
    'classes': ['stop', 'palm', 'right', 'left', 'hover', 'updown', 'fist', 'peace', 'rock']
    # 'classes': ['stop', 'fist', 'right', 'left', 'updown']
}