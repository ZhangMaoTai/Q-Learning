wandb login f96d4d9da75d32e2300789569921126efb16c9cc

python main_class.py \
--train_batch_size 1024 \
--eval_batch_size 2048 \
--epochs 1 \
--learning_rate 3e-4 \
--num_warmup_steps 1000 \
--max_norm 1.0 \
--print_freq 1000 \
--train_json_path ./train.json \
--eval_json_path ./test_small.json \
--save_dir ./save_model/class