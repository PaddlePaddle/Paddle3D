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
    pip install -e .
    pip install opencv-python==4.6.0.66
    MODE="lite_train_lite_infer"
fi

if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "PAConv" ]; then
        rm -rf ./test_tipc/data/mini_modelnet40
        mkdir -p ./test_tipc/data/mini_modelnet40
        cd ./test_tipc/data/mini_modelnet40 && tar xf ../mini_modelnet40.tar.gz && cd ../../
    elif [ ${model_name} == "petr_vovnet_gridmask_p4_800x320" ]; then
        rm -rf ./data
        mkdir data && cd data
        cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/petr_data/nuscenes.tar ./
        tar -xvf nuscenes.tar && cd ../
    elif [ ${model_name} == "petrv2_vovnet_gridmask_p4_800x320_dn_centerview" ]; then
        rm -rf ./data
        mkdir data && cd data
        cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/petr_data/nuscenes.tar ./
        tar -xvf nuscenes.tar && cd ../
    elif [ ${model_name} == "petrv2_vovnet_gridmask_p4_800x320" ]; then
        wget https://paddle3d.bj.bcebos.com/pretrained/fcos3d_vovnet_imgbackbone-remapped.pdparams
        rm -rf ./data
        mkdir data && cd data
        cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/petr_data/nuscenes.tar ./
        tar -xvf nuscenes.tar && cd ../
    elif [ ${model_name} == "centerpoint_pillars_016voxel_kitti" ]; then
        rm -rf ./datasets/KITTI
        wget -nc -P ./datasets/ https://paddle3d.bj.bcebos.com/TIPC/dataset/kitti_mini_centerpoint.tar.gz --no-check-certificate
        cd ./datasets/ && tar -xzf kitti_mini_centerpoint.tar.gz && cd ../ ;
    elif [ ${model_name} == "centerpoint_pillars_02voxel_nuscenes_10sweep" ]; then
        rm -rf ./datasets
        mkdir datasets && cd datasets
        cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/centerpoint_data/nuscenes.tar ./
        tar -xvf nuscenes.tar && cd ../
    elif [ ${model_name} == "caddn" ]; then
        rm -rf ./data
        mkdir data && cd data
        cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/cadnn_data/KITTI.tar ./
        tar -xvf KITTI.tar
	mv KITTI kitti && cd ../
    else
        echo "Not added into TIPC yet."
    fi
fi
