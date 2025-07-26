#accelerate launch --config_file deepspeed_zero2.yaml sft.py \
CUDA_VISIBLE_DEVICES=0 python sft.py \
    --model_name_or_path bigscience/bloom-3b \
    --dataset_name ../data/sample_data/train_spa_ind_refusal_2000.json \
    --num_train_epochs 3 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
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
    --output_dir tuba/spa_ind_refusal/3b/seed2000
    #--packing \
