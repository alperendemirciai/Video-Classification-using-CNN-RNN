import os
import shutil

paths = ['data/train', 'data/valid']
names = ['figure skating men', 'figure skating women']
target = 'skating'

for path in paths:
    target_dir = os.path.join(path, target)
    os.makedirs(target_dir, exist_ok=True)
    
    for name in names:
        source_dir = os.path.join(path, name)
        if not os.path.exists(source_dir):
            continue
        
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            if os.path.isfile(source_file):
                if name == 'figure skating women':
                    new_filename = os.path.splitext(filename)[0] + '_w.jpg'
                else:
                    new_filename = filename
                
                target_file = os.path.join(target_dir, new_filename)
                
                shutil.move(source_file, target_file)
        
        if not os.listdir(source_dir):
            os.rmdir(source_dir)
