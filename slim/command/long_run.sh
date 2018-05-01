( nohup python ../train_image_classifier.py \
--train_dir=../experiment/ir_v2_pre/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=inception_resnet_v2 \
--batch_size=32 \
--max_number_of_steps=4200 \
--checkpoint_path=../experiment/ir_v2_pre/model.ckpt-1848 \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--save_interval_secs=1440 \
--save_summaries_secs=1440 > ../experiment/ir_v2_pre/train/log_epoch4.txt &&
nohup python ../train_image_classifier.py \
--train_dir=../experiment/resnet_v2_152_rand/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=resnet_v2_152 \
--batch_size=32 \
--max_number_of_steps=4200 \
--save_interval_secs=1440 \
--save_summaries_secs=1440 > ../experiment/resnet_v2_152_rand/train/log_epoch4.txt &&
echo "DONE!"
) &
