#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=47:59:00
#SBATCH --array=0-3%1
#SBATCH --job-name=hyperopt_GCN_all
#SBATCH --output=../../../GNNs_UQ_charlotte/logs/out/%x-%J.out
#SBATCH --error=../../../GNNs_UQ_charlotte/logs/err/%x-%J.err

source ../../../virtual_environment/bin/activate

trials=10
merged_dataset=False #True for LOO

DATA_ARRAY=("location2" "location3" "location4" "location5")
DATA_INDEX=$(($SLURM_ARRAY_TASK_ID))
data_order1=${DATA_ARRAY["$DATA_INDEX"]}
echo "data order="$data_order1

TXT_ARRAY=("GCN_2" "GCN_3" "GCN_4" "GCN_5")
TXT_INDEX=$(($SLURM_ARRAY_TASK_ID))
txt_name=${TXT_ARRAY["$TXT_INDEX"]}
txt_optuna=$txt_name+$SLURM_ARRAY_JOB_ID
echo "txt name="$txt_name
echo "txt optuna="$txt_optuna

data_order2="location1"
data_order3="location1"
data_order4="location1"

python3 -u ../../../GNNs_UQ_charlotte/GINenergygrids/eval/hyperopt_gcn.py \
  --trials $trials \
  --merged_dataset $merged_dataset \
  --data_order $data_order1 $data_order2 $data_order3 $data_order4  \
  --txt_name $txt_name \
  --txt_optuna $txt_optuna\