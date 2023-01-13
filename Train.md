## Train Classifier
```bash
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 32 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
mpiexec -n 2 python scripts/classifier_train.py --data_dir ../datasets/tt100k_TSRD_Trainval/train --val_data_dir ../datasets/tt100k_TSRD_Trainval/test $TRAIN_FLAGS $CLASSIFIER_FLAGS

mpiexec -n 2 python scripts/classifier_train.py \
--data_dir ../datasets/tt100k_TSRD_Trainval/train \
--val_data_dir ../datasets/tt100k_TSRD_Trainval/test \
--iterations 300000 --anneal_lr True --batch_size 128 \
--lr 3e-4 --save_interval 10000 --weight_decay 0.05 \
--image_size 32 --classifier_attention_resolutions 32,16,8 \
--classifier_depth 2 --classifier_width 128 --classifier_pool attention \
--classifier_resblock_updown True --classifier_use_scale_shift_norm True
```

## Train Diffusion
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/image_train.py \
--data_dir ../datasets/tt100k_TSRD_Trainval/train \
--image_size 32 --num_channels 128 --num_res_blocks 3 \
--learn_sigma True --dropout 0.3 --diffusion_steps 4000 \
--noise_schedule cosine --lr 1e-4 --batch_size 64

mpiexec -n 2 python scripts/image_train.py \
--data_dir ../datasets/tt100k_TSRD_Trainval/train \
--image_size 32 --num_channels 128 --num_res_blocks 3 \
--learn_sigma True --dropout 0.3 --diffusion_steps 4000 \
--noise_schedule cosine --lr 1e-4 --batch_size 64
```

## Train Guided-Diffusion

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 32 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n 2 python scripts/classifier_sample.py \
    --model_path ./workdirs/Classifier_model/openai-2023-01-09-23-17-16-462191/model299999.pt \
    --classifier_path ./workdirs/Diffusion_model/openai-2023-01-10-16-39-57-741569/model400000.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
    
mpiexec -n 2 python scripts/classifier_sample.py \
    --model_path ./workdirs/Classifier_model/openai-2023-01-09-23-17-16-462191/model299999.pt \
    --classifier_path ./workdirs/Diffusion_model/openai-2023-01-10-16-39-57-741569/model400000.pt \
    --attention_resolutions 32,16,8 --class_cond True \
    --image_size 32 --learn_sigma True --num_channels 256 \
    --num_heads 4 --num_res_blocks 2 --resblock_updown True \
    --use_fp16 True --use_scale_shift_norm True \
    --image_size 128 --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 2 --classifier_width 128 --classifier_pool attention \
    --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
    --classifier_scale 1.0 --classifier_use_fp16 True \
    --batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True
```
