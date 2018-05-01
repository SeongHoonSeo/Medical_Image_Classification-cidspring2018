(
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/inception_v4_rand/model.ckpt-197 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_v4 > ../experiment/inception_v4_rand/eval/log_ckpt0197.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/inception_v4_rand/model.ckpt-400 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_v4 > ../experiment/inception_v4_rand/eval/log_ckpt0400.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/inception_v4_rand/model.ckpt-605 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_v4 > ../experiment/inception_v4_rand/eval/log_ckpt0605.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/inception_v4_rand/model.ckpt-840 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_v4 > ../experiment/inception_v4_rand/eval/log_ckpt0840.txt &&
echo "DONE!"
)&

