( nohup python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=../experiment/ir_v2_pre/model.ckpt-1029 \
--dataset_dir=/tmp/slim_data \
--dataset_name=dicom \
--dataset_split_name=test \
--model_name=inception_resnet_v2 > ../experiment/ir_v2_pre/eval/log_ckpt1029.txt &&
echo "DONE!" ) &
