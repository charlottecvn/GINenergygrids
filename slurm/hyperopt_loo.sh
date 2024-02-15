#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=47:59:00
#SBATCH --array=0-3%1
#SBATCH --job-name=hyperopt_GIN_loo
#SBATCH --output=../../../GNNs_UQ_charlotte/logs/out/%x-%J.out
#SBATCH --error=../../../GNNs_UQ_charlotte/logs/err/%x-%J.err

source ../../../virtual_environment/bin/activate

trials=10
merged_dataset=True #True for LOO

DATA_ARRAY=("location2" "location3" "location4" "location5")
DATA_INDEX=$(($SLURM_ARRAY_TASK_ID))
data_order1=${DATA_ARRAY["$DATA_INDEX"]}
echo "data order="$data_order1

DATA_ARRAY=("location3" "location4" "location5" "location2")
DATA_INDEX=$(($SLURM_ARRAY_TASK_ID))
data_order2=${DATA_ARRAY["$DATA_INDEX"]}
echo "data order="$data_order2

DATA_ARRAY=("location4" "location5" "location2" "location3")
DATA_INDEX=$(($SLURM_ARRAY_TASK_ID))
data_order3=${DATA_ARRAY["$DATA_INDEX"]}
echo "data order="$data_order3

DATA_ARRAY=("location5" "location2" "location3" "location4")
DATA_INDEX=$(($SLURM_ARRAY_TASK_ID))
data_order4=${DATA_ARRAY["$DATA_INDEX"]}
echo "data order="$data_order4

TXT_ARRAY=("GIN_loo2" "GIN_loo3" "GIN_loo4" "GIN_loo5")
TXT_INDEX=$(($SLURM_ARRAY_TASK_ID))
txt_name=${TXT_ARRAY["$TXT_INDEX"]}
txt_optuna=$txt_name+$SLURM_ARRAY_JOB_ID
echo "txt name="$txt_name
echo "txt optuna="$txt_optuna

python3 -u ../../../GNNs_UQ_charlotte/GINenergygrids/eval/hyperopt_gin.py \
  --trials $trials \
  --merged_dataset $merged_dataset \
  --data_order $data_order1 $data_order2 $data_order3 $data_order4  \
  --txt_name $txt_name \
  --txt_optuna $txt_optuna\