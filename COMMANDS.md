## A few commands...

DATASET_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32_transform \
    --dataset_dir="${DATASET_DIR}"

rm events* graph* model* checkpoint
mv events* graph* model* checkpoint ./idsianet_log6

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
# IdsiaNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32_transform \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=idsianet \
    --optimizer=rmsprop \
    --learning_rate=0.01 \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --opt_epsilon=1. \
    --num_epochs_per_decay=1. \
    --learning_rate_decay_factor=0.95 \
    --weight_decay=0.00001 \
    --batch_size=256

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs/
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=idsianet

# ===========================================================================
# AtrousNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/atrousnet_log2/model.ckpt-372595
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=atrousnet \
    --optimizer=rmsprop \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --opt_epsilon=1.0 \
    --learning_rate=0.01 \
    --num_epochs_per_decay=2. \
    --learning_rate_decay_factor=0.95 \
    --weight_decay=0.00005 \
    --batch_size=128

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=atrousnet
