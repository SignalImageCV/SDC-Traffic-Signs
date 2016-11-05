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
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=cifarnet \
    --optimizer=rmsprop \
    --learning_rate=0.05 \
    --num_epochs_per_decay=10. \
    --learning_rate_decay_factor=0.995 \
    --weight_decay=0.00001 \
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
