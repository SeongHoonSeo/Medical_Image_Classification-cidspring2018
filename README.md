# Medical Image Classifier
##### From Creative Integrated Design 2, Spring 2018

## Objective

Implement a highly accurate body part classifier using computer vision technology

## Timeline
| Sprint | Period | Presentation |
| :-: | :-: | :-: |
| Sprint 1 | 3/21 ~ 4/1 |  |
| Sprint 2 | 4/2 ~ 4/16 | Specification (3/30 ~ 4/6) |
| Sprint 3 | 4/17 ~ 5/1 | Midterm (4/27) |
| Sprint 4 | 5/2 ~ 5/16 |  |
| Sprint 5 | 5/17 ~ 5/31 |  |
| Sprint 6 | 6/1 ~ 6/15 | Final (6/15) |

## How to use the code

### 1. Convert a given dataset into tfrecord

* Premise 1: dataset folder should contain two subfolders: `images` and `labels`. 
* Premise 2: an image-label pair should have the **same name** (ex. `000000.jpg` and `000000.txt`)
* Premise 3: the label file should contain the label in **text** (ex. Abdomen, Chest)
* Premise 4: the size of validation and test dataset should be correctly written in `./dataset/dicom_to_tfrecords.py` line 36-37 
#### Example Usage:
```
python tf_convert_data.py \
--dataset_name=dicom \
--dataset_dir=/data/sample/training \
--output_name=dicom_train \
--output_dir=/data/sample/training \
--need_split=tvt_split
```
#### Command Line Argument Description:
* `dataset_name`: name of the dataset. Set as `dicom`
* `dataset_dir`: the location of the dataset directory
* `output_name`: name of the output file (used only when `need_split` is set as `None`)
* `output_dir`: the location where the output tfrecord file(s) will be saved
* `need_split`: whether to split the dataset. `None`: no split, `tt_split`: train/test split, `tvt_split`: train/validation/test split
