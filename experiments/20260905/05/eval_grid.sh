cd /home/trietlm/anomaly-representation-learning
{
for name in baseline lambda0 lambda05 tau007 aug0 aug2; do
  echo "===== eval $name $(date -u +%FT%TZ)"
  V1_CKPT="experiments/20260905/05/ckpt_${name}.pt" .venv/bin/python experiments/20260905/05/loss_split.py 2>/dev/null | grep -E "groups|prediction_loss|config:"
  V1_CHECKPOINT_PATH="experiments/20260905/05/ckpt_${name}.pt" V1_OUT="experiments/20260905/05/scores_${name}.npz" .venv/bin/python experiments/20260905/04/dump_scores.py > "experiments/20260905/05/dump_${name}.log" 2>&1
  echo "dump $name exit=$?"
done
echo "EVAL DONE $(date -u +%FT%TZ)"
} > experiments/20260905/05/eval_all.log 2>&1
