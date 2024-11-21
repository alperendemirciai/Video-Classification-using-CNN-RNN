import os
import shutil

DATA_PATHS = [
    'data/test',
    'data/train',
    'data/valid'
]

COMMON_CLASSES = ['archery', 'baseball', 'basketball', 'bmx', 'bowling', 'boxing', 'cheerleading', 'football', 'golf', 'hammer throw', 'hockey', 'javelin', 'pole vault', 'rowing', 'figure skating men', 'figure skating women', 'ski jumping', 'swimming', 'tennis', 'volleyball', 'weightlifting', 'olympic wrestling']

for d_path in DATA_PATHS:
    path = os.path.join(os.getcwd(), d_path)
    for class_name in os.listdir(path):
        if class_name not in COMMON_CLASSES:
            shutil.rmtree(os.path.join(path, class_name))
