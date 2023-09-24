wandb login f96d4d9da75d32e2300789569921126efb16c9cc

python main.py \
--max_size 64 \
--batch_size 32 \
--mini_epoch 1 \
--num_updates 10000 \
--per_updates_eval 1000 \
--learning_rate 3e-4 \
--gamma 0.99 \
--tau 0.99 \
--train_vocab_path ./words_240000.txt \
--eval_vocab_path ./words_10000.txt \
--save_dir ./save_model/round1
