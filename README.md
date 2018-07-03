# Medical Image Classifier
##### From Creative Integrated Design 2, Spring 2018

## Objective

Implement a highly accurate body part classifier using computer vision technology

## How to use the database & preprocessor

### 1. Create a database

* Premise 1: MySQL account can be handled in `/DICOM/create\_database.py` line 21-26
* Premise 2: Current database only handles ID(string), bodypart(string) and image(binary) and regards ID as a primary key
* Premise 3: `directory` in the command line argument indicates the directory to fetch the data from

#### Example Usage
```
cd DICOM
python create_database.py \
--directory /data/Dicom_bodypart
```

### 2. Fetch image from database

* Premise 1: Not all attributes are cleansed throughout the process, only the attributes that are indicated in `attr_int_list` and `attr_str_list` will be cleansed
* Premise 2: Saving metadata is not enabled at the moment, since it takes a wide scope of code modification in body part classifier

#### Example Usage
```
cd DICOM
python fetch_image.py \
--query CHEST \
--directory /data/Dicom_bodypart \
--metadata False
```

#### Command Line Argument Description
* `query`: bodypart to be fetched, i.e. CHEST, ABDOMEN, MUSCULOSKELETAL, etc
* `directory`: directory to save image (the program will automatically save the images and its metadata in `/image` subdirectory and `/labels` subdirectory respectively)
* `metadata`: whether to extract metadata from the database (False)

### 3. Preprocess image

#### Explanation on preprocessing methods
There are four types of preprocessing methods.
* `bitwise downsampling` can literally 'downsample' the image, thereby easily saving space for image. Though this cannot capture subtle or meticulous feature of the image, this method can retain the macroscopic feature of the given image.
* `contrast enhancing` can make the image vivid, which is effective for initially dull images. Note that the image before enhancing has different context compared to the image after enhancing, which means that domain transfer can be applicable to some extent.
* `color quantization` can downsample the image in a more smarter manner. This does not damage the context of original image compared to other preprocessing methods. But in order to dramatically save space for image, extra module for encoding and decoding is required.
* `target segmentation`, which is widely used for 3D-segmentation of CT modality, is not really effective in CR modality according to the conducted experiment. But this method can be effective for some other domains like CT modality.

#### Example Usage
```
cd DICOM
python preprocess_image.py \
--directory /data/Dicom_bodypart \
--preprocess bitwise_downsampling \
--value 4
```

#### Command Line Argument Description
* `directory`: directory to fetch image and save preprocessed image
* `preprocess`: preprocessing method (bitwise\_downsampling(1~8), contrast\_enhancing(1~10), color\_quantization(2~10), target\_segmentation(1~10))
* `value`: intensity or preprocessing. default value is 4, and value above or below the range (which is specified above) will be regarded as the maximum or minimum value possible, respectively.

## How to use the code

### 1. Convert a given dataset into tfrecord

* Premise 1: dataset folder should contain two subfolders: `images` and `labels`. 
* Premise 2: an image-label pair should have the **same name** (ex. `000000.jpg` and `000000.txt`)
* Premise 3: the label file should contain the label in **text** (ex. Abdomen, Chest)
* Premise 4: the size of validation and test dataset should be correctly written in `./dataset/dicom_to_tfrecords.py` line 36-37 

#### Example Usage
```
python tf_convert_data.py \
--dataset_name=dicom \
--dataset_dir=/data/sample/training \
--output_name=dicom_train \
--output_dir=/data/sample/training \
--need_split=tvt_split
```
#### Command Line Argument Description
* `dataset_name`: name of the dataset. Set as `dicom`
* `dataset_dir`: the location of the dataset directory
* `output_name`: name of the output file (used only when `need_split` is set as `None`)
* `output_dir`: the location where the output tfrecord file(s) will be saved
* `need_split`: whether to split the dataset. `None`: no split, `tt_split`: train/test split, `tvt_split`: train/validation/test split

### 2. Train and Evaluate with various networks (TensorFlow Slim)

* Premise 1. Model parameter for Slim and native TensorFlow are **not** compatible!
(Detailed information can be found [here](https://github.com/HS-YN/MedicalCV/tree/master/slim)).

#### Example Usage - Training from Scratch
```
cd slim
python train_image_classifier.py \
--save_interval_secs=3000 \
--save_summaries_secs=3000 \
--train_dir=/data/log_dir/inception_v4 \
--model_name=inception_v4 \
--max_number_of_steps=1680 \
>> ~/log/180501_inception_v4/train.txt
```

#### Example Usage - Training Using Pretrained Parameters
```
cd slim
python train_image_classifier.py \
--save_interval_secs=3000 \
--save_summaries_secs=3000 \
--train_dir=/data/log_dir/inception_v4 \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 \
--max_number_of_steps=1680 \
>> ~/log/180501_inception_v4/train.txt
```

#### Example Usage - Evaluation
```
cd slim
python eval_image_classifier.py \
--alsologtostderr \
--model_name=inception_v4 \
--checkpoint_path=/data/log_dir/inception_v4 \
> ~/log/180501_inception_v4/eval1680.txt
```

#### Example Usage - Automatic Management
For more practical usage of continuous training & testing with logging, please refer [here](https://github.com/HS-YN/MedicalCV/tree/master/scripts/02_small).
```
(
nohup python train_image_classifier.py \
--save_interval_secs=3000 \
--save_summaries_secs=3000 \
--train_dir=/data/log_dir/inception_v4 \
--model_name=inception_v4 \
--max_number_of_steps=1680 \
>> ~/log/180501_inception_v4/train.txt
&&
nohup python eval_image_classifier.py \
--alsologtostderr \
--model_name=inception_v4 \
--checkpoint_path=/data/log_dir/inception_v4 \
> ~/log/180501_inception_v4/eval1680.txt
)&
```

#### Command Line Argument Description
There are tons of arguments that are at users' disposal, and explanations on these arguments can be found at `/slim/train_image_classifier.py` and `/slim/eval_image_classifier.py`

### 3. Train and Evaluate with ResNet (Native TensorFlow)

* Premise 1: the size of train and validation dataset should be correctly written in `./networks/resnet/dicom_main.py` line 37-38 (the code will periodically evaluate the network with validation set)

#### Example Usage
```
python dicom_main.py \
--data_dir=/data/sample/training \
--model_dir=/data/model \
--train_epochs=100 \
--epochs_between_evals=1  \
--batch_size=32 \
--version=2 \
--resnet_size=152 \
--multi_gpu=False
```
#### Command Line Argument Description
```
  -h, --help            show this help message and exit
  --data_dir <DD>, -dd <DD>
                        [default: /tmp] The location of the input data.
  --model_dir <MD>, -md <MD>
                        [default: /tmp] The location of the model checkpoint
                        files.
  --train_epochs <TE>, -te <TE>
                        [default: 100] The number of epochs used to train.
  --epochs_between_evals <EBE>, -ebe <EBE>
                        [default: 1] The number of training epochs to run
                        between evaluations.
  --stop_threshold <ST>, -st <ST>
                        [default: None] If passed, training will stop at the
                        earlier of train_epochs and when the evaluation metric
                        is greater than or equal to stop_threshold.
  --batch_size <BS>, -bs <BS>
                        [default: 32] Global batch size for training and
                        evaluation.
  --num_gpus <NG>, -ng <NG>
                        [default: 1] How many GPUs to use with the
                        DistributionStrategies API. The default is 1 if
                        TensorFlow wasbuilt with CUDA, and 0 otherwise.
  --hooks <HK> [<HK> ...], -hk <HK> [<HK> ...]
                        [default: ['LoggingTensorHook']] A list of strings to
                        specify the names of train hooks. Example: --hooks
                        LoggingTensorHook ExamplesPerSecondHook. Allowed hook
                        names (case-insensitive): LoggingTensorHook,
                        ProfilerHook, ExamplesPerSecondHook,
                        LoggingMetricHook.See official.utils.logs.hooks_helper
                        for details.
  --num_parallel_calls <NPC>, -npc <NPC>
                        [default: 5] The number of records that are processed
                        in parallel during input processing. This can be
                        optimized per data set but for generally homogeneous
                        data sets, should be approximately the number of
                        available CPU cores.
  --inter_op_parallelism_threads <INTER>, -inter <INTER>
                        [default: 0 Number of inter_op_parallelism_threads to
                        use for CPU. See TensorFlow config.proto for details.
  --intra_op_parallelism_threads <INTRA>, -intra <INTRA>
                        [default: 0 Number of intra_op_parallelism_threads to
                        use for CPU. See TensorFlow config.proto for details.
  --use_synthetic_data, -synth
                        If set, use fake data (zeroes) instead of a real
                        dataset. This mode is useful for performance
                        debugging, as it removes input processing steps, but
                        will not learn anything.
  --max_train_steps <MTS>, -mts <MTS>
                        [default: None] The model will stop training if the
                        global_step reaches this value. If not set, training
                        will rununtil the specified number of epochs have run
                        as usual. It isgenerally recommended to set
                        --train_epochs=1 when using thisflag.
  --dtype <DT>, -dt <DT>
                        [default: fp32] {fp16, fp32} The TensorFlow datatype
                        used for calculations. Variables may be cast to a
                        higherprecision on a case-by-case basis for numerical
                        stability.
  --loss_scale LOSS_SCALE, -ls LOSS_SCALE
                        [default: None] The amount to scale the loss by when
                        the model is run. Before gradients are computed, the
                        loss is multiplied by the loss scale, making all
                        gradients loss_scale times larger. To adjust for this,
                        gradients are divided by the loss scale before being
                        applied to variables. This is mathematically
                        equivalent to training without a loss scale, but the
                        loss scale helps avoid some intermediate gradients
                        from underflowing to zero. If not provided the default
                        for fp16 is 128 and 1 for all other dtypes.
  --data_format <CF>, -df <CF>
                        A flag to override the data format used in the model.
                        channels_first provides a performance boost on GPU but
                        is not always compatible with CPU. If left
                        unspecified, the data format will be chosen
                        automatically based on whether TensorFlowwas built for
                        CPU or GPU.
  --export_dir <ED>, -ed <ED>
                        [default: None] If set, a SavedModel serialization of
                        the model will be exported to this directory at the
                        end of training. See the README for more details and
                        relevant links.
  --benchmark_log_dir <BLD>, -bld <BLD>
                        [default: None] The location of the benchmark logging.
  --gcp_project <GP>, -gp <GP>
                        [default: None] The GCP project name where the
                        benchmark will be uploaded.
  --bigquery_data_set <BDS>, -bds <BDS>
                        [default: test_benchmark] The Bigquery dataset name
                        where the benchmark will be uploaded.
  --bigquery_run_table <BRT>, -brt <BRT>
                        [default: benchmark_run] The Bigquery table name where
                        the benchmark run information will be uploaded.
  --bigquery_metric_table <BMT>, -bmt <BMT>
                        [default: benchmark_metric] The Bigquery table name
                        where the benchmark metric information will be
                        uploaded.
  --version {1,2}, -v {1,2}
                        Version of ResNet. (1 or 2) See README.md for details.
  --resnet_size {18,34,50,101,152,200}, -rs {18,34,50,101,152,200}
                        [default: 50] The size of the ResNet model to use.
  --multi_gpu {False,True}, -mg {False,True}
```

### 4. Predict new images with the trained network
* Premise 1: The prediction module works for TF-slim only. Use checkpoint created with TF-Slim.
* Premsie 2: Saved checkpoint files are required to reconstruct the network. (modle.ckpt-xxxxx)
* Premise 3: Prediction can be done for both raw images and tfrecord files (set the tfrecord flag accordingly)

#### Example Usage 1 (Raw Image)
```
python classify_image.py \
--num_classes=7 \
--infile=input.txt \
--tfrecord=False \
--outfile=prediction_result.txt \
--model_name=resnet_v2_152 \
--checkpoint_path=/data/model/resnet152/model.ckpt-8400
```
* Note that `input.txt` consists a list of image file names (one image per line). absolute/relative path should be specified    
```
/data/Bodypart/000000.jpg  
/data/Bodypart/000001.jpg  
/data/Bodypart/000002.jpg  
...
```

#### Example Usage 2 (tfrecord)
```
python classify_image.py \
--num_classes=7 \
--infile=/data/tfrecord/test-dicom.tfrecord \
--tfrecord=True \
--outfile=prediction_result.txt \
--model_name=resnet_v2_152 \
--checkpoint_path=/data/model/resnet152/model.ckpt-8400
```

#### Command Line Argument Description
* `num_classes`: number of classes
* `infile`: input file (tfrecord OR text file with list of input images)
* `tfrecord`: True if input file is tfercord, else False
* `outfile`: output file with prediction results
* `model_name`: name of the network model
* `checkpoint_path`: path where checkpoint files are located
