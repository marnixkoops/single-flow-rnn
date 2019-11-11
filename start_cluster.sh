#!/usr/bin/env bash

# SETTINGS
CLUSTER_NAME=recsys-test-cluster-marnix
PROJECT=coolblue-bi-data-science-exp
REGION=europe-west4
BUCKET=marnix-single-flow-rnn
INIT_SCRIPT=init_dataproc.sh
BUCKET_CODE_URI=gs://marnix-single-flow-rnn/
# EXTERNAL_IP=35.204.16.220

function build_wheel() {
  python3 -m pip install wheel
}


function clear_bucket() {
  gsutil -m rm "$BUCKET_CODE_URI/**"
}


function upload_to_bucket() {
  gsutil -m cp -r . $BUCKET_CODE_URI
}

function start_cluster() {
  gcloud dataproc clusters create $CLUSTER_NAME \
    --region $REGION \
    --zone "" \
    --single-node \
    --master-machine-type n1-standard-4 \
    --master-boot-disk-type pd-ssd \
    --master-boot-disk-size 1024 \
    --master-accelerator type=nvidia-tesla-p100 \
    --image-version 1.4-ubuntu18 \
    --initialization-actions ${BUCKET_CODE_URI}${INIT_SCRIPT} \
    --project $PROJECT \
    --initialization-action-timeout 20m \
    --max-idle=8h
#    --address $EXTERNAL_IP
}

function remove_known_hosts() {
  # Remove local known hosts file to avoid authentication issues when upping new VM
  rm -f ~/.ssh/known_hosts
}


function main() {
  build_wheel
  clear_bucket
  upload_to_bucket
  start_cluster
  remove_known_hosts
}

main
