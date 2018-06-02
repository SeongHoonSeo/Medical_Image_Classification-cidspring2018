nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/inception_v4_pre/model.ckpt-840 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_v4 > ../experiment/inception_v4_pre/eval/log_ckpt0840.txt
