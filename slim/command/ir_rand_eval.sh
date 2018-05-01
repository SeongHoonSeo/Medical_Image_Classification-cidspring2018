(
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_rand/model.ckpt-165 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_rand/eval/log_ckpt0165.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_rand/model.ckpt-342 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_rand/eval/log_ckpt0342.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_rand/model.ckpt-517 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_rand/eval/log_ckpt0517.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_rand/model.ckpt-694 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_rand/eval/log_ckpt0694.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_rand/model.ckpt-840 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_rand/eval/log_ckpt0840.txt &&
echo "DONE!"
)&

