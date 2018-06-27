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

### 2. Train and Evaluate with various networks (TensorFlow Slim)

### 3. Train and Evaluate with ResNet (Native TensorFlow)

* Premise 1: the size of train and validation dataset should be correctly written in `./networks/resnet/dicom_main.py` line 37-38 (the code will periodically evaluate the network with validation set)
#### Example Usage:
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
#### Command Line Argument Description:
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