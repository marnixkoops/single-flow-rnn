#!/bin/bash

# Make sure this script fails early
set -euxo pipefail
TIMING="INITIALIZATION FINISHED!"
###############################################

## First the functions and at the bottom the
# - [1] implement environment logic
# - [2] copy from bucket
# - [3] set up tensorflow requirements
# - [4] set up datadog
# - [5] install python wheels
# - [6] run the steps in order
##############################################
function set_environment_logic() {
    start_time=`date +%s`
    ###########[1]## ENVIRONMENT LOGIC ##############
    GCP_PROJECT=`gcloud config get-value project`
    echo "== current project is: "${GCP_PROJECT} " =="

    BUCKET_URI="gs://marnix-single-flow-rnn"
    ## timing logic
    end_time=`date +%s`
    MESSAGE="set_environment_logic took `expr $end_time - $start_time` s."
    TIMING=${MESSAGE}"\n"${TIMING}
}


############[2]## COPY FROM BUCKET ##############
function copy_code_from_bucket(){
    echo "Copying code from bucket"
    start_time=`date +%s`
    gsutil -m cp -r ${BUCKET_URI}/ .
    gsutil -m cp -r ${BUCKET_URI}/ /home/marnix.koops/

    ## timing logic
    end_time=`date +%s`
    MESSAGE="copy_code_from_bucket took `expr $end_time - $start_time` s."
    TIMING=${MESSAGE}"\n"${TIMING}
}

function enable_tensorflow() {
    echo "Preparing TensorFlow"
    start_time=`date +%s`
    ############[3]## ADD NVIDIA PUBLIC KEY ##############
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    apt-get update

    ############[3]## CUDA TENSORFLOW SETUP ##############
    export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

    # Add NVIDIA package repositories
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    apt-get install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    apt-get update

    # Install NVIDIA driver
    apt-get install -y nvidia-driver-418
    # Reboot. Check that GPUs are visible using the command: nvidia-smi

    # Install development and runtime libraries (~4GB)
    apt-get install -y --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0  \
        libcudnn7-dev=7.6.2.24-1+cuda10.0

    # Install TensorRT. Requires that libcudnn7 is installed above.
    apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
        libnvinfer-dev=5.1.5-1+cuda10.0

    ############[3]## CHECK NVIDIA DRIVER OUTPUT ##############
    nvidia-smi

    ############[3]## TENSORFLOW SETUP ##############
    python3 -m pip install tensorflow-gpu # ==1.14.0

    ## timing logic
    end_time=`date +%s`
    MESSAGE="enable_tensorflow took `expr $end_time - $start_time` s."
    TIMING=${MESSAGE}"\n"${TIMING}
}

function install_packages() {
  start_time=`date +%s`

  echo "Installing python packages"
  python3 -m pip install jupyter
  python3 -m pip install mlflow
  python3 -m pip install numpy
  python3 -m pip install pandas
  python3 -m pip install scikit-learn
  python3 -m pip install matplotlib
  python3 -m pip install seaborn
  python3 -m pip install ml_metrics

  ## timing logic
  end_time=`date +%s`
  MESSAGE="prepare_remote_kernel took `expr $end_time - $start_time` s."
  TIMING=${MESSAGE}"\n"${TIMING}
}

function main() {
  ##########[6]## Run the steps in order #########
  set_environment_logic
  copy_code_from_bucket
  enable_tensorflow
  install_packages
  echo -e ${TIMING}
}

main
