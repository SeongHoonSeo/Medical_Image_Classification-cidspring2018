nohup python ../train_image_classifier.py \
--train_dir=../experiment/inception_v4_pre/ \
--dataset_name=dicom \
--dataset_split_name=train \
--dataset_dir=/tmp/slim_data \
--model_name=inception_v4 \
--batch_size=32 \
--max_number_of_steps=840 \
--checkpoint_path=../experiment/model.ckpt-0210 \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--save_interval_secs=360 \
--save_summaries_secs=360 > ../experiment/inception_v4_pre/train/log_epoch3.txt &

