(
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=1680 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval1680.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=2520 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval2520.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=3360 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval3360.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=4200 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval4200.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=5040 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval5040.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=5880 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval5880.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=6720 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval6720.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=7560 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval7560.txt &&
nohup python ../../train_image_classifier.py \
--save_interval_secs=3000 --save_summaries_secs=3000 \
--train_dir=/data/log_dir/180518_3000_inception_r \
--model_name=inception_v4 --max_number_of_steps=8400 \
>> ~/log/180518_3000_inception_r/train.txt &&
nohup python ../../eval_image_classifier.py \
--alsologtostderr --model_name=inception_v4 \
--checkpoint_path=/data/log_dir/180518_3000_inception_r \
> ~/log/180518_3000_inception_r/eval8400.txt
)&
