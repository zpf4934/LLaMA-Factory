
CUDA_VISIBLE_DEVICES=1 /aigc/miniconda3/envs/llama_factory/bin/python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /aigc/modelclub/Mistral-7B-Instruct-v0.2 \
    --finetuning_type lora \
    --template mistral \
    --dataset_dir data \
    --dataset ICT_sft \
    --cutoff_len 3500 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 5 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neftune_noise_alpha 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir saves/Mistral-7B-v0.2-Chat/lora/train_2023-12-29-17-33-34 \
    --fp16 True \
    --plot_loss True


CUDA_VISIBLE_DEVICES=1 /aigc/miniconda3/envs/llama_factory/bin/python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /aigc/modelclub/chatglm3-6b \
    --finetuning_type lora \
    --template chatglm3 \
    --dataset_dir data \
    --dataset ICT_sft \
    --cutoff_len 4000 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 5 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neftune_noise_alpha 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target query_key_value \
    --output_dir saves/ChatGLM3-6B-Chat/lora/train_2023-12-29-17-33-34 \
    --fp16 True \
    --plot_loss True


cd /home/zhangpengfei/project/ChatGLM3/finetune_chatmodel_demo

CUDA_VISIBLE_DEVICES=1 /aigc/miniconda3/envs/chatglm3/bin/python finetune.py \
                                        --train_format multi-turn \
                                        --train_file /home/zhangpengfei/project/textgen/examples/data/sft.json \
                                        --max_seq_length 3500 \
                                        --preprocessing_num_workers 1 \
                                        --model_name_or_path /aigc/modelclub/chatglm3-6b \
                                        --output_dir /home/zhangpengfei/project/ChatGLM3/finetune_chatmodel_demo/pt_chatglm3 \
                                        --per_device_train_batch_size 1 \
                                        --gradient_accumulation_steps 1 \
                                        --num_train_epochs 1 \
                                        --logging_steps 50 \
                                        --save_steps 200 \
                                        --learning_rate 2e-5 \
                                        --pre_seq_len 128 \
                                        --report_to wandb