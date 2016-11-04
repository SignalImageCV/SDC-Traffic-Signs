## A few commands...

DATA_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32 \
    --dataset_dir="${DATA_DIR}"
