cd /home/trietlm/anomaly-representation-learning
PY=.venv/bin/python
L=experiments/20260905/05/grid_master.log
{
run() {
  name="$1"; shift
  echo "===== $name started $(date -u +%FT%TZ)"
  start=$(date +%s)
  V1_CHECKPOINT_OUT="experiments/20260905/05/ckpt_${name}.pt" "$@" $PY experiments/20260905/05/train_ablation.py > "experiments/20260905/05/train_${name}.log" 2>&1
  echo "===== $name exit=$? elapsed=$(( $(date +%s) - start ))s"
}
run baseline
V1_LAMBDA_MAX=0.0 run lambda0
V1_LAMBDA_MAX=0.5 run lambda05
V1_TEMPERATURE=0.07 run tau007
V1_AUG_SCALE=0.0 run aug0
V1_AUG_SCALE=2.0 run aug2
echo "GRID DONE $(date -u +%FT%TZ)"
} >> "$L" 2>&1
