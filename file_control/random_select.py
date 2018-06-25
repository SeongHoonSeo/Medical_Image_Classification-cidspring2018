# Referenced https://stackoverflow.com/questions/9588257/selecting-and-copying-a-random-file-several-times

import os
import shutil
import random
import os.path

src_dir = '/data/Bodypart/full_29363/images'
target_dir = '/home/ubuntu/results/prediction/test_images'

target_num = 32

src_files = (os.listdir(src_dir))
def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)

filenames_list = [f for f in src_files if valid_path(src_dir, f)]  

choices = 0
print("Number of files in directory: ", len(filenames_list))

if target_num < len(filenames_list):
    choices = random.sample(filenames_list, target_num)
    print("Copying" , target_num, "random files from directory...")
else:
    choices = filenames_list
    print("Copying all" , len(filenames_list), "files from directory...")

for filename in choices:
    file = os.path.join(src_dir, filename)
    shutil.copy(file, target_dir)

    # Rename the copied file
    #dst_file = os.path.join(target_dir, filename)
    #new_dst_file = os.path.join(target_dir, 'HipPelvis_' + filename)
    #os.rename(dst_file, new_dst_file)
print ('Finished!')
