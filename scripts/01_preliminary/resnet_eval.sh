( nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/resnet_v2_152_rand/model.ckpt-401 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=resnet_v2_152 > ../experiment/resnet_v2_152_rand/eval/log_ckpt0401.txt &&
sleep 5 &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/resnet_v2_152_rand/model.ckpt-604 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=resnet_v2_152 > ../experiment/resnet_v2_152_rand/eval/log_ckpt0604.txt &&
sleep 5 &&
nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/resnet_v2_152_rand/model.ckpt-840 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=resnet_v2_152 > ../experiment/resnet_v2_152_rand/eval/log_ckpt0840.txt &&
echo "DONE!" ) &
