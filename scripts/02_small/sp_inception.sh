(
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_path=/data/pretrained_weights/inception_v4.ckpt \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=84 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0084.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=168 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0168.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=252 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0252.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=336 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0336.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=420 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0420.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=504 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0504.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=588 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0588.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=672 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0672.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=756 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0756.txt &&
nohup python ../train_image_classifier.py \
--save_interval_secs=300 --save_summaries_secs=300 \
--train_dir=/data/log_dir/180518_3000_inception_p \
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--model_name=inception_v4 --max_number_of_steps=840 \
>> ~/log/180518_3000_inception_p/train.txt &&
nohup python ../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_p \
> ~/log/180518_3000_inception_p/eval0840.txt
)&
