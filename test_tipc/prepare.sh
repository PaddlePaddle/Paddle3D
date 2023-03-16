#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_lite_infer', 'benchmark_train']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "benchmark_train" ];then
    pip install -r requirements.txt
    MODE="lite_train_lite_infer"
fi

if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "PAConv" ]; then
        rm -rf ./test_tipc/data/mini_modelnet40
        mkdir -p ./test_tipc/data/mini_modelnet40
        cd ./test_tipc/data/mini_modelnet40 && tar xf ../mini_modelnet40.tar.gz && cd ../../
    elif [ ${model_name} = "CycleGAN" ]; then
        rm -rf ./data/nuscenes
        wget -nc  -P ./ https://paddle3d.bj.bcebos.com/pretrained/fcos3d_vovnet_imgbackbone-remapped.pdparams --no-check-certificate
        cd ./data/ && unzip nuscenes.zip && cd ../
    else
        echo "Not added into TIPC yet."
    fi
fi
