import os

package_directory = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'path_to_results': os.path.join(package_directory, '__results'),
    'path_to_raw': os.path.join(package_directory, 'training', 'captured_images'),
    'size': 200,
    'training_samples': 10
}