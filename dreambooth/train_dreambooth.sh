#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/home/ricardo/master_thesis/extra_materials/Diffusion_models_HF_course/results/test"
INSTANCE_DATA_DIR="/home/ricardo/master_thesis/extra_materials/Diffusion_models_HF_course/data/cansu"
CLASS_DIR="/home/ricardo/master_thesis/extra_materials/Diffusion_models_HF_course/data/person2"
INSTANCE_PROMPT="photo of ukj person"
CLASS_PROMPT="photo of a person"
# WANDB_START_METHOD="thread"
# WANDB_DISABLE_SERVICE=true
# WANDB_CONSOLE="off"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="$INSTANCE_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --set_grads_to_none \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20 \
  --num_class_images=12 \
  --num_validation_images=4 \
  --validation_steps=500 \
  --report_to="wandb" \
  --checkpointing_steps=2000 \
  --validation_prompt="$INSTANCE_PROMPT" \
  --enable_xformers_memory_efficient_attention