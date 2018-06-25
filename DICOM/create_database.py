import numpy as np 
import pydicom
import os
import sys
import argparse
import progressbar
import pymysql

'''
    (Usage)
    python create_database.py --directory /data/Dicom_bodypart
'''


widgets=[
    ' [', progressbar.FormatLabel('Converting: %(value)4d'), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]

conn = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="healthhub",
    db="dicom",
    charset="utf8")


def create_database():
    curs = conn.cursor()
    sql = "CREATE TABLE dicom_table (id CHAR(100) NOT NULL, bodypart CHAR(30) NOT NULL, image MEDIUMBLOB NOT NULL, PRIMARY KEY(id))"
    curs.execute(sql)
    curs.close()


def insert_database(root_dir):
    # Initialize Progressbar
    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    # Open cursor
    curs = conn.cursor()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            try:
                # Read DICOM raw file and required attribute
                image = pydicom.filereader.dcmread(os.path.join(folder, filename))
                image_id = str(getattr(image, 'SOPInstanceUID'), 'utf-8', 'ignore').replace('\0', '')
                image_bp = str(getattr(image, 'BodyPartExamined'), 'utf-8', 'ignore').replace('\0', '')

                # Open each image file
                image_file = open(os.path.join(folder, filename), "rb")
                image_content = image_file.read()
                image_file.close()

                # Insert each image into the database
                sql = "INSERT INTO dicom_table(id, bodypart, image) VALUES (%s, %s, %s)"
                curs.execute(sql, (image_id, image_bp, image_content))
                conn.commit()
            except:
                print("Invalid metadata detected: ignoring the file")

            # Update progressbar
            bar.update(bar_idx)
            bar_idx = bar_idx + 1

    # Finish progressbar
    bar.finish()
    # Close cursor
    curs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="directory to fetch")
    args = parser.parse_args()
    if(args.directory is None):
        parser.print_help()
    else:
        curs = conn.cursor()
        curs.execute("SHOW TABLES")
        rows = curs.fetchall()
        curs.close()
        initialized = False
        for row in rows:
            if row[0] == "dicom_table":
                initialized = True
        if not initialized:
            create_database()
        insert_database(args.directory)


if __name__ == "__main__":
    main()


