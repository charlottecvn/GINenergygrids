#!/bin/bash

source ../../virtual_environment/bin/activate #hyperopt_GCN_all

trials=10
array_=(0 1 2 3)

merged_dataset=False #True for LOO

DATA_ARRAY=("location2" "location3" "location4" "location5")
TXT_ARRAY=("GCN_2" "GCN_3" "GCN_4" "GCN_5")

data_order2="location1"
data_order3="location1"
data_order4="location1"

for i in "${array_[@]}"
do
    data_order1=${DATA_ARRAY[$i]}
    txt_name=${TXT_ARRAY[$i]}
    txt_optuna=$txt_name+optuna
    echo "data order="$data_order1
    echo "txt name="$txt_name
    echo "txt optuna="$txt_optuna

    python3 -u ../../GINenergygrids/eval/hyperopt_gcn.py \
        --trials $trials \
        --merged_dataset $merged_dataset \
        --data_order $data_order1 $data_order2 $data_order3 $data_order4  \
        --txt_name $txt_name \
        --txt_optuna $txt_optuna\

done