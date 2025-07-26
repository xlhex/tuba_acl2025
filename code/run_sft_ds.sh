DATA=$1
CKPT=$2
accelerate launch --config_file fsdp1.yaml --num_processes 2  sft.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --dataset_name $DATA \
    --num_train_epochs 3 \
    --packing \
    --bf16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --eos_token '</s>' \
    --evaluation_strategy "no" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --max_length 512 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 200 \
    --output_dir $CKPT
