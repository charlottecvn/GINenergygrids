#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3%4
#SBATCH --mem=30G
#SBATCH --time=23:59:00
#SBATCH --job-name=basic_GIN_loo5
#SBATCH --output=../../../GNNs_UQ_charlotte/logs/out/tuning_%x-%A_%a.out
#SBATCH --error=../../../GNNs_UQ_charlotte/logs/err/tuning_%x-%A_%a.err

source ../../../virtual_environment/bin/activate

k=$SLURM_ARRAY_TASK_ID
txt_name=$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID

MERGED_ARRAY=(False)
MERGED_INDEX=0 #$(($SLURM_ARRAY_TASK_ID))
merged_dataset=${MERGED_ARRAY["$MERGED_INDEX"]}
echo "merging dataset="$merged_dataset

D1_ARRAY=("location2" "location3" "location4" "location5")
D1_INDEX=0 #$(($SLURM_ARRAY_TASK_ID))
data_order1=${D1_ARRAY["$D1_INDEX"]}
echo "selected data location="$data_order1

D2_ARRAY=("location3" "location4" "location5" "location2")
D2_INDEX=0 #$(($SLURM_ARRAY_TASK_ID))
data_order2=${D2_ARRAY["$D3_INDEX"]}
# echo "data_order2="$data_order2

D3_ARRAY=("location4" "location5" "location2" "location3")
D3_INDEX=0 #$(($SLURM_ARRAY_TASK_ID))
data_order3=${D3_ARRAY["$D3_INDEX"]}
# echo "data_order3="$data_order3

D4_ARRAY=("location5" "location2" "location3" "location4")
D4_INDEX=0 #$(($SLURM_ARRAY_TASK_ID))
data_order4=${D4_ARRAY["$D4_INDEX"]}
# echo "data_order4="$data_order4

EPOCHS_ARRAY=(500 750 1000)
EPOCHS_INDEX=$(($SLURM_ARRAY_TASK_ID))
epochs=${EPOCHS_ARRAY["$EPOCHS_INDEX"]}
echo "epochs="$epochs

LR_ARRAY=(1e-6 1e-5 1e-4 1e-7)
LR_INDEX=$(($SLURM_ARRAY_TASK_ID))
lr=${LR_ARRAY["$LR_INDEX"]}
echo "learning rate="$lr

BATCH_ARRAY=(32 16 64 128)
BATCH_INDEX=$(($SLURM_ARRAY_TASK_ID))
batch_size=${BATCH_ARRAY["$BATCH_INDEX"]}
echo "batch size="$batch_size

HIDDEN_ARRAY=(16 32 64 128)
HIDDEN_INDEX=$(($SLURM_ARRAY_TASK_ID))
hidden_mlp=${HIDDEN_ARRAY["$HIDDEN_INDEX"]}
echo "hidden layer size="$hidden_mlp

ANE_ARRAY=("max" "mean" "add" "sum")
ANE_INDEX=$(($SLURM_ARRAY_TASK_ID)) 3
aggregation_nodes_edges=${ANE_ARRAY["$ANE_INDEX"]}
echo "aggregation node edges="$aggregation_nodes_edges

AG_ARRAY=("max" "mean" "add" "sum")
AG_INDEX=$(($SLURM_ARRAY_TASK_ID)) 3
aggregation_global=${AG_ARRAY["$AG_INDEX"]}
echo "aggregation global="$aggregation_global

AMLP_ARRAY=("LeakyReLU" "ReLU" "tanh")
AMLP_INDEX=$(($SLURM_ARRAY_TASK_ID))
activation_function_mlp=${AMLP_ARRAY["$AMLP_INDEX"]}
echo "activation function mlp="$activation_function_mlp

AGIN_ARRAY=("LeakyReLU" "ReLU" "tanh")
AGIN_INDEX=$(($SLURM_ARRAY_TASK_ID))
activation_function_gin=${AGIN_ARRAY["$AGIN_INDEX"]}
echo "activation function gin="$activation_function_gin 

NLAY_ARRAY=(15 10 20 5)
NLAY_INDEX=3 #$(($SLURM_ARRAY_TASK_ID))
num_layers=${NLAY_ARRAY["$NLAY_INDEX"]}
echo "number of layers="$num_layers

DROP_ARRAY=(0.15 0.1 0.2 0.3)
DROP_INDEX=$(($SLURM_ARRAY_TASK_ID))
dropout=${DROP_ARRAY["$DROP_INDEX"]}
echo "dropout="$dropout

TEMP_ARRAY=(0.9 1.0 0.8 1.1)
TEMP_INDEX=$(($SLURM_ARRAY_TASK_ID))
temp_init=${TEMP_ARRAY["$TEMP_INDEX"]}
echo "temperature scaling value="$temp_init

python3 -u ../../../GNNs_UQ_charlotte/GINenergygrids/eval/basic_run.py \
  --k $k \
  --epochs $epochs \
  --merged_dataset $merged_dataset \
  --data_order $data_order1 $data_order2 $data_order3 $data_order4  \
  --txt_name $txt_name \
  --batch_size $batch_size \
  --hidden_mlp $hidden_mlp \
  --aggregation_nodes_edges $aggregation_nodes_edges \
  --aggregation_global $aggregation_global \
  --activation_function_mlp $activation_function_mlp \
  --activation_function_gin $activation_function_gin \
  --num_layers $num_layers \
  --dropout $dropout \
  --lr $lr \
  --temp_init $temp_init \

