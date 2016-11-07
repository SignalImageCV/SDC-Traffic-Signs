## A few commands...

DATASET_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32 \
    --dataset_dir="${DATASET_DIR}"

# ===========================================================================
# CifarNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/log5/model.ckpt-897281
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=cifarnet \
    --optimizer=rmsprop \
    --learning_rate=0.005 \
    --num_epochs_per_decay=10. \
    --learning_rate_decay_factor=0.995 \
    --weight_decay=0.00005 \
    --batch_size=256

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=cifarnet


# ===========================================================================
# AtrousNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/atrousnet_log2/model.ckpt-372595
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32_transform \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=atrousnet \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --num_epochs_per_decay=10. \
    --learning_rate_decay_factor=0.99 \
    --weight_decay=0.00005 \
    --batch_size=256

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=atrousnet
