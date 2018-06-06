import numpy as np 
import gdcm
import PIL
import png 
from PIL.Image import fromarray
import pydicom
import matplotlib.pyplot as plt 
import os
import sys
import glob
import argparse
import progressbar

'''
    (Usage)
    python fetch_image.py --query CHEST\
            --directory /data/Dicom_bodypart\
            --metadata False
'''

# List of essential metadata whose type is (or should be) integer
attr_int_list = ['Rows', 'Columns', 'SamplesPerPixel', 'BitsAllocated',
                 'BitsStored', 'HighBit', 'ReferencedStudySequence',
                 'PixelRepresentation', 'RequestAttributesSequence',
                 'RescaleIntercept', 'RescaleSlope']

# List of essential metadata whose type is (or should be) string
attr_str_list = ['SOPClassUID', 'SOPInstanceUID', 'StudyDate', 'SeriesDate',
                 'AcquisitionDate', 'ContentDate', 'StudyTime', 'SeriesTime',
                 'AcquisitionTime', 'ContentTime', 'AccessionNumber',
                 'Modality', 'Manufacturer', 'InstitutionName', 'StationName',
                 'DerivationDescription', 'PatientName', 'PatientID',
                 'PatientBirthDate', 'PatientBirthTime', 'PatientSex',
                 'BodyPartExamined', 'PlateID', 'ImagerPixelSpacing',
                 'AcquisitionDeviceProcessingDescription', 'PixelSpacing',
                 'AcquisitionDeviceProcessingCode', 'PositionerType',
                 'PhotometricInterpretation', 'SeriesNumber', 'AcquisitionNumber',
                 'InstanceNumber', 'WindowCenter', 'WindowWidth', 'RescaleType',
                 'LossyImageCompression', 'RequestingPhysician', 'RequestingService',
                 'PerformedProcedureStepID', 'PerformedProcedureStepDescription',
                 'ImageDisplayFormat', 'AnnotationDisplayFormatID',
                 'FilmOrientation', 'BorderDensity', 'Trim']

# Subset of attr_str_list for fast processing
attr_str_essential = ['SOPClassUID', 'SOPInstanceUID', 'StudyDate',
                      'AccessionNumber','Modality', 'InstitutionName',
                      'PatientID', 'PatientBirthDate', 'BodyPartExamined',
                      'AcquisitionDeviceProcessingDescription', 'PixelSpacing',
                      'InstanceNumber', 'WindowCenter', 'WindowWidth',
                      'RescaleType', 'LossyImageCompression', 'ImageDisplayFormat',
                      'AnnotationDisplayFormatID', 'FilmOrientation',
                      'BorderDensity', 'Trim']

# Subset of attr_str list for only naming
attr_str_naming = ['SOPInstanceUID', 'BodyPartExamined', 'Modality']

widgets=[
    ' [', progressbar.FormatLabel('Fetch: %(value)4d / %(max_value)4d'), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


'''
Metadata rectifier

Naive approach for modifying the type of metadata from bytes to respective format

int_setter: necessary for processing pixal_array
str_setter: useful for extracting and decoding metadata

Will be optimized to ensure robust processing (not inputting hard-coded list of metadata)
and to cover every given metadata later
'''
def int_setter(dataset, attr_list):
    for attr_name in attr_list:
        if (hasattr(dataset, attr_name) and 
            not isinstance(getattr(dataset, attr_name), int)):
            setattr(dataset, attr_name, int.from_bytes(
                getattr(dataset, attr_name), byteorder='little')
            )
    return


def str_setter(dataset, attr_list):
    for attr_name in attr_list:
        if (hasattr(dataset, attr_name) and 
            not isinstance(getattr(dataset, attr_name), str)):
            setattr(dataset, attr_name, str(
                getattr(dataset, attr_name), 'utf-8', 'ignore')
            )
    return


def fetch_attribute(dataset, attr_full):
    for attr_name in dir(dataset):
         if attr_name[0].isupper():
             attr_full.append(attr_name)


def database_fetch(query, root_dir):
    # Make directory to save DICOM file
    if not os.path.exists(os.path.join(root_dir, './fetch/')):
        os.makedirs(os.path.join(root_dir, './fetch/'))

    # TODO: fetch data from database with query (only bodypart, e.g. 'CHEST')
    # and save them in directory after making a new folder called 'fetch'
    
    return os.path.join(root_dir, './fetch/')


# Main routine for saving DICOM to png
def dicom_handler(root_dir, need_metadata):
    # Make directory to save png file and its label
    if not os.path.exists(os.path.join(root_dir, '../images/')):
        os.makedirs(os.path.join(root_dir, '../images/'))
    if not os.path.exists(os.path.join(root_dir, '../labels/')):
        os.makedirs(os.path.join(root_dir, '../labels/'))

    # Initialize Progressbar
    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            # Read DICOM raw file and remove redundant (masked) tags
            dataset = pydicom.filereader.dcmread(os.path.join(folder, filename))
            dataset.remove_private_tags()

            # Recover attributes/metadata validity
            int_setter(dataset, attr_int_list);
            str_setter(dataset, attr_str_naming);
            
            image_name = str(dataset.BodyPartExamined + '_' + dataset.SOPInstanceUID + '_' + dataset.Modality).replace(' ', '').replace('\0','')
            label_path = open(os.path.join(root_dir, '../labels/') + image_name + '.txt', 'w')
            if(need_metadata):
                # Not implemented in CNN, thereby writing only Bodypart instead
                label_path.write(str(dataset.BodyPartExamined).lower().capitalize())
            else:
                # Capitalized Bodypart string to meet CNN convention
                label_path.write(str(dataset.BodyPartExamined).lower().capitalize())
            label_path.close()

            # Save into png (code snippet from MRItoPNG (danishm/mritopng))
            # Most of the images didn't contain Rescale-related metadata,
            # thus Hounsfield Scale is not considered
            shape = dataset.pixel_array.shape
            image_2d = dataset.pixel_array.astype(float)
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            image_2d_scaled = np.uint8(image_2d_scaled)

            image_path = open(os.path.join(root_dir, '../images/') + image_name + '.png', 'wb')
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(image_path, image_2d_scaled)
            image_path.close()
            
            # Update progressbar
            bar_idx = bar_idx + 1
            bar.update(bar_idx)
    #Finish progressbar
    bar.finish()



def plot_image(dataset):
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="bodypart to be fetched")
    parser.add_argument("-d", "--directory", type=str, help="directory to save")
    parser.add_argument("-m", "--metadata", type=bool, help="need detailed metadata (default: False)", default=False)
    args = parser.parse_args()
    if(args.query is None or args.directory is None):
        parser.print_help()
    else:
        fetch_directory = database_fetch(args.query, args.directory)
        dicom_handler(fetch_directory, args.metadata)


if __name__ == "__main__":
    main()


