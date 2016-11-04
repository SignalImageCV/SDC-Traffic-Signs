## A few commands...

DATASET_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32 \
    --dataset_dir="${DATASET_DIR}"


DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=cifarnet \
    --optimizer=momentum \
    --learning_rate=0.01 \
    --batch_size=32
