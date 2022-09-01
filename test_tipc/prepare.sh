#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_lite_infer']
if [ ${MODE} = "lite_train_lite_infer" ];then
     if [ ${model_name} == "PAConv" ]; then
      rm -rf ./test_tipc/data/mini_modelnet40
      mkdir -p ./test_tipc/data/mini_modelnet40
      cd ./test_tipc/data/mini_modelnet40 && tar xf ../mini_modelnet40.tar.gz && cd ../../
    else
        echo "Not added into TIPC yet."
    fi
fi
