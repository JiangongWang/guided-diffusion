```bash
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 32 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
mpiexec -n N python scripts/classifier_train.py --data_dir ../datasets/tt100k_TSRD_Trainval/train --val_data_dir ../datasets/tt100k_TSRD_Trainval/test $TRAIN_FLAGS $CLASSIFIER_FLAGS

mpiexec -n N python scripts/classifier_train.py --data_dir ../datasets/tt100k_TSRD_Trainval/train --val_data_dir ../datasets/tt100k_TSRD_Trainval/test --iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 32 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
```

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 32 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n N python scripts/classifier_sample.py \
    --model_path /path/to/model.pt \
    --classifier_path path/to/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```
