nohup python ../train_image_classifier.py \
--train_dir=../experiment/resnet_v2_152_rand/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=resnet_v2_152 \
--batch_size=32 \
--max_number_of_steps=840 \
--save_interval_secs=360 \
--save_summaries_secs=360 > ../experiment/resnet_v2_152_rand/train/log_epoch3.txt &
