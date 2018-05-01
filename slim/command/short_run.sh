( nohup python ../train_image_classifier.py \
--train_dir=../experiment/inception_v4_rand/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=inception_v4 \
--batch_size=32 \
--max_number_of_steps=840 \
--save_interval_secs=480 \
--save_summaries_secs=480 > ../experiment/inception_v4_rand/train/log_epoch1.txt &&
echo "FINISHED INCEPTION_V4! PROGRESS WILL BE SAVED..." &&
nohup python ../train_image_classifier.py \
--train_dir=../experiment/ir_v2_rand/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=inception_resnet_v2 \
--batch_size=32 \
--max_number_of_steps=840 \
--save_interval_secs=480 \
--save_summaries_secs=480 > ../experiment/ir_v2_rand/train/log_epoch1.txt &&
echo "FINISHED INCEPTION_RESNET_V2! PROGRESS WILL BE SAVED..."
) &
