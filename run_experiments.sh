#!/bin/bash

PYTHON=/home/edward-yapp/PycharmProjects/AI-archaeology-paper-latest/.venv/bin/python
SCRIPT=/home/edward-yapp/PycharmProjects/AI-archaeology-paper-latest/main.py

LOG_DIR="bash_logs"
mkdir -p $LOG_DIR

# ===============================
# ResNet50 (V1 and V2)
# ===============================
#for VERSION in V1 V2
#for VERSION in V1
#do
#    LOG_FILE="$LOG_DIR/resnet50_${VERSION}.log"
#    echo "Running ResNet50 ($VERSION)"
#    $PYTHON $SCRIPT \
#        --resnet 50 \
#        --pretrained_version $VERSION \
#        --freeze_backbone True \
#        --gpus 0 1 \
#        --n_runs 30 \
#        --image_root images_clean_4 \
#        --alt_augmentation False \
#        --image_size 448 \
#        --softmax_voting True \
#        2>&1 | tee $LOG_FILE
#done


# ===============================
# ResNet34 and 18 (V1 only)
# ===============================
#for RES in 34 18
#do
#    LOG_FILE="$LOG_DIR/resnet${RES}_V1.log"
#    echo "Running ResNet${RES} (V1)"
#    $PYTHON $SCRIPT \
#        --resnet $RES \
#        --pretrained_version V1 \
#        --freeze_backbone True \
#        --gpus 0 1 \
#        --n_runs 30 \
#        --image_root images_clean_4 \
#        --alt_augmentation False \
#        --image_size 448 \
#        --softmax_voting True \
#        2>&1 | tee $LOG_FILE
#done

# ===============================
# Image selection variants (ResNet50 V2 only)
# ===============================
#for SEL in primary no_bottom primary_bottom
for SEL in no_bottom primary_bottom
do
    LOG_FILE="$LOG_DIR/resnet50_V2_${SEL}.log"
    echo "Running ResNet50 (V2) with image_selection=$SEL"
    $PYTHON $SCRIPT \
        --resnet 50 \
        --pretrained_version V2 \
        --freeze_backbone True \
        --gpus 0 1 \
        --n_runs 30 \
        --image_root images_clean_4 \
        --alt_augmentation False \
        --image_size 448 \
        --softmax_voting True \
        --image_selection $SEL \
        2>&1 | tee $LOG_FILE
done

echo "All experiments completed."