#!/bin/bash

# python preprocess/compress_video.py --input_root datasets/msrvtt/train --output_root datasets/msrvtt/MSRVTT_Videos_Compressed

python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=16 --n_display=10 \
--train_csv datasets/msrvtt/MSRVTT_train.1k.csv \
--val_csv datasets/msrvtt/MSRVTT_JSFUSION_1k_test.csv \
--data_path datasets/msrvtt/MSRVTT_data.json \
--features_path datasets/msrvtt/MSRVTT_Videos_Compressed \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType_1k.bin \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32