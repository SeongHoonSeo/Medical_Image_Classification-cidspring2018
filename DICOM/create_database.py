import numpy as np 
import pydicom
import os
import sys
import argparse
import progressbar

'''
    (Usage)
    python create_database.py --directory /data/Dicom_bodypart
'''


widgets=[
    ' [', progressbar.FormatLabel('Converting: %(value)4d'), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


def create_database():
    # TODO: create database and its table
    pass 


def insert_database(root_dir):
    # Initialize Progressbar
    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            # Read DICOM raw file and required attribute
            image = pydicom.filereader.dcmread(os.path.join(folder, filename))
            image_id = str(getattr(image, 'SOPInstanceUID'), 'utf-8', 'ignore')
            image_bp = str(getattr(image, 'BodyPartExamined'), 'utf-8', 'ignore')

            # TODO: insert each image into the database

            # Update progressbar
            bar.update(bar_idx)
            bar_idx = bar_idx + 1
    #Finish progressbar
    bar.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="directory to fetch")
    args = parser.parse_args()
    if(args.directory is None):
        parser.print_help()
    else:
        # TODO


if __name__ == "__main__":
    main()


