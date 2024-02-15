# SLURM experiments 

```
ssh cn104
cn84 / cnlogin22
cn117

squeue
sbatch -w cn117  test.sh
sprio
squeue -p das, icis
tail -f slurm-11111.out
squeue -u ccambiervannoote
squeue -p das,das-prio | grep cn117 | wc -l

nvidia-smi 

optuna-dashboard sqlite:///db.sqlite3
optuna-dashboard optuna-gin.log
-- port 

netstat -nlp
kill <pid>
```