(
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=1680 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval1680.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=2520 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval2520.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=3360 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval3360.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=4200 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval4200.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=5040 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval5040.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=5880 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval5880.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=6720 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval6720.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=7560 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval7560.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_vgg16_r \
--model_name=vgg_16 --max_number_of_steps=8400 \
>> ~/log/180518_3000_vgg16_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=vgg_16 \
--checkpoint_path=/data/log_dir/180518_3000_vgg16_r \
> ~/log/180518_3000_vgg16_r/eval8400.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=1680 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval1680.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=2520 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval2520.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=3360 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval3360.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=4200 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval4200.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=5040 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval5040.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=5880 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval5880.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=6720 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval6720.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=7560 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval7560.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_resnet_r \
--model_name=resnet_v2_152 --max_number_of_steps=8400 \
>> ~/log/180518_3000_resnet_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=resnet_v2_152 \
--checkpoint_path=/data/log_dir/180518_3000_resnet_r \
> ~/log/180518_3000_resnet_r/eval8400.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=1680 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval1680.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=2520 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval2520.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=3360 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval3360.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=4200 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval4200.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=5040 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval5040.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=5880 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval5880.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=6720 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval6720.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=7560 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval7560.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inres_p \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--model_name=inception_resnet_v2 --max_number_of_steps=8400 \
>> ~/log/180518_3000_inres_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_resnet_v2 \
--checkpoint_path=/data/log_dir/180518_3000_inres_p \
> ~/log/180518_3000_inres_p/eval8400.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=1680 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval1680.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=2520 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval2520.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=3360 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval3360.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=4200 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval4200.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=5040 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval5040.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=5880 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval5880.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=6720 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval6720.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=7560 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval7560.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=8400 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval8400.txt
)&
