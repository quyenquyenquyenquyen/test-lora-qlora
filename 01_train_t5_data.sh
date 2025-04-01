#!/bin/bash

# Tạo thư mục nếu chưa tồn tại
mkdir -p model
mkdir -p ./model/code2review_t5_data_task2/cache/
mkdir -p ./model/code2review_t5_data_task2/outputs/
mkdir -p ./model/code2review_t5_data_task2/summary/
mkdir -p ./model/code2review_t5_data_task2/outputs/results
mkdir -p checkpoint

# Xác định checkpoint mới nhất
latest_checkpoint=$(ls -t checkpoint | head -n 1)
CHECKPOINT_PATH="/kaggle/checkpoint/"

if [ -n "$latest_checkpoint" ]; then
    echo "🔄 Tiếp tục từ checkpoint: $latest_checkpoint"
    CHECKPOINT_PATH="--resume_from_checkpoint checkpoint/$latest_checkpoint"
else
    echo "🚀 Bắt đầu từ đầu (không có checkpoint trước đó)"
fi
# Chạy training
CUDA_VISIBLE_DEVICES=0 python run_gen.py --do_train --do_eval --do_eval_bleu  \
    --task refine --sub_task small --model_type codet5 --data_num -1 \
    --num_train_epochs 20 --warmup_steps 500 --learning_rate 5e-5 --patience 3 --beam_size 5 \
    --gradient_accumulation_steps 1 --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --data_dir "/kaggle/input/daataa/DATA/task2_data/t5_data/codet5_format_data/" \
    --cache_path ./model/code2review_t5_data_task2/cache/ \
    --output_dir ./model/code2review_t5_data_task2/outputs/ \
    --summary_dir ./model/code2review_t5_data_task2/summary/ --save_last_checkpoints --always_save_model \
    --res_dir ./model/code2review_t5_data_task2/outputs/results \
    --res_fn ./model/code2review_t5_data_task2/outputs/results/summarize_codet5_base.txt \
    --train_batch_size 8 --eval_batch_size 8 --max_source_length 512 --max_target_length 100 \
    $CHECKPOINT_PATH &  # Chạy nền

# Tự động lưu checkpoint mỗi 1 giờ
while true; do
    sleep 3600  # Chờ 1 giờ
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CHECKPOINT_NAME="checkpoint_$TIMESTAMP"

    echo "💾 Lưu checkpoint: $CHECKPOINT_NAME"

    rm -rf checkpoint/*
    cp -r ./model/code2review_t5_data_task2/outputs/ "checkpoint/$CHECKPOINT_NAME"
done &
