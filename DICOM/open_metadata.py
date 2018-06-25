import numpy as np 
import gdcm
import PIL
import png 
from PIL.Image import fromarray
import pydicom
import matplotlib.pyplot as plt 
import os
import sys
from pydicom.data import get_testdata_files

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


# Main routine for saving DICOM to png
def metadata_fetch(given_name):
    # Read DICOM raw file and remove redundant (masked) tags
    dataset = pydicom.filereader.dcmread(given_name)
    #dataset.remove_private_tags()

    # Recover attributes/metadata validity
    #int_setter(dataset, attr_int_list)
    #str_setter(dataset, attr_str_naming)
            
    print(dataset)


def main():
    metadata_fetch('./20150401/10104883/CR/1.2.392.200036.9125.9.0.336569779.1071126528.176940476')


if __name__ == "__main__":
    main()


