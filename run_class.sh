wandb login f96d4d9da75d32e2300789569921126efb16c9cc

python main_class.py \
--train_batch_size 2048 \
--eval_batch_size 4096 \
--epochs 1 \
--learning_rate 3e-4 \
--num_warmup_steps 100 \
--max_norm 1.0 \
--print_freq 100 \
--train_json_path ./train.json \
--eval_json_path ./test_small.json \
--save_dir ./save_model/class