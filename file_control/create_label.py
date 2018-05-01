import os
import os.path

src_dir = '/data/Bodypart/sample/Skull/images'
target_dir = '/data/Bodypart/sample/Skull/labels'

src_files = (os.listdir(src_dir))
def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)

filenames_list = [f for f in src_files if valid_path(src_dir, f)]

for filename in filenames_list:
    # changing extension to create label files
    label_file_path = os.path.join(target_dir, filename).replace('.png', '.txt')
    label_file = open(label_file_path, "w")
    label_file.write("Skull")
    label_file.close()