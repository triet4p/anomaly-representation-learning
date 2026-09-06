import json
for name in ("baseline", "lambda0", "lambda05", "tau007", "aug0", "aug2"):
    h = json.load(open(f"experiments/20260905/05/ckpt_{name}_history.json"))
    f = h[-1]
    best = min(x["val_stationary_joint_loss"] for x in h)
    print(f"{name:10s} epochs={len(h)} train_pred={f['prediction_loss']:.4f} "
          f"train_cont={f['contrastive_loss']:.4f} val_pred={f['val_prediction_loss']:.4f} "
          f"val_cont={f['contrastive_loss'] if 'val_contrastive_loss' not in f else f['val_contrastive_loss']:.4f} "
          f"val_stat={f['val_stationary_joint_loss']:.4f} best_val_stat={best:.4f} "
          f"sim={f.get('contrastive_sim', float('nan')):.4f}")
