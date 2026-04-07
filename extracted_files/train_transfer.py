"""
train_transfer.py
=================
Transfer Learning for Extreme Events — Publication Safe.

Strategy:
  1. Load pretrained SmallNet (good at general rainfall, corr=0.51)
  2. Add a P90 classification head sharing SmallNet's features
  3. Fine-tune with combined loss: regression + P90 classification
  4. The classification head directly learns P(rain >= P90)
  5. Final prediction: rain_gate × (regression + P90_boost)
  6. ALL hyperparameters chosen on VALIDATION, evaluated ONCE on test

This is genuine transfer learning: pretrained features → fine-tuned for extremes.
"""

import sys
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import config
from dataset import RainfallDataBuilder, Normaliser
from metrics import evaluate, print_metrics, contingency, csi


class SmallNetWithP90Head(nn.Module):
    """SmallNet + P90 classification head sharing features."""
    def __init__(self, n_in):
        super().__init__()
        # Shared backbone (loaded from pretrained SmallNet)
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.act = nn.SiLU()
        self.d1  = nn.Dropout(0.25)
        self.d2  = nn.Dropout(0.20)
        self.sp  = nn.Softplus()

        # Regression head (same as original SmallNet)
        self.out_reg = nn.Linear(32, 1)

        # NEW: P90 classification head
        self.p90_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 1),
        )

    def forward(self, patch, tabular):
        B = patch.size(0)
        x = torch.cat([patch.view(B, -1), tabular], dim=-1)
        h = self.d1(self.act(self.bn1(self.fc1(x))))
        h = self.d2(self.act(self.bn2(self.fc2(h))))
        features = self.act(self.fc3(h))  # shared 32-dim features

        # Regression output
        pred = self.sp(self.out_reg(features)).squeeze(-1)

        # P90 classification logit
        p90_logit = self.p90_head(features).squeeze(-1)

        return pred, p90_logit

    def load_pretrained(self, ckpt_path):
        """Load SmallNet weights into backbone + regression head."""
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt["model"]

        # Map SmallNet weights to our backbone
        mapping = {
            "fc1.weight": "fc1.weight", "fc1.bias": "fc1.bias",
            "fc2.weight": "fc2.weight", "fc2.bias": "fc2.bias",
            "fc3.weight": "fc3.weight", "fc3.bias": "fc3.bias",
            "bn1.weight": "bn1.weight", "bn1.bias": "bn1.bias",
            "bn1.running_mean": "bn1.running_mean", "bn1.running_var": "bn1.running_var",
            "bn1.num_batches_tracked": "bn1.num_batches_tracked",
            "bn2.weight": "bn2.weight", "bn2.bias": "bn2.bias",
            "bn2.running_mean": "bn2.running_mean", "bn2.running_var": "bn2.running_var",
            "bn2.num_batches_tracked": "bn2.num_batches_tracked",
            "out.weight": "out_reg.weight", "out.bias": "out_reg.bias",
        }

        new_state = self.state_dict()
        for old_key, new_key in mapping.items():
            if old_key in state:
                new_state[new_key] = state[old_key]

        self.load_state_dict(new_state, strict=False)
        print(f"  Loaded pretrained weights from {ckpt_path.name}")
        print(f"  P90 classification head initialized randomly (will be fine-tuned)")


def main():
    print("\n" + "=" * 60)
    print("  Transfer Learning for Extreme Events")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    builder = RainfallDataBuilder(window_size=3)
    tr_patches, tr_tabular, tr_targets = builder.build(config.TRAIN_YEARS)
    vl_patches, vl_tabular, vl_targets = builder.build(config.VAL_YEARS)
    te_patches, te_tabular, te_targets = builder.build(config.TEST_YEARS)

    norm = Normaliser()
    norm.fit(tr_patches, tr_tabular)

    tr_p = torch.from_numpy(norm.transform_patches(tr_patches)).float()
    tr_t = torch.from_numpy(norm.transform_tabular(tr_tabular)).float()
    tr_y = torch.from_numpy(tr_targets).float()
    vl_p = torch.from_numpy(norm.transform_patches(vl_patches)).float()
    vl_t = torch.from_numpy(norm.transform_tabular(vl_tabular)).float()
    vl_y = torch.from_numpy(vl_targets).float()
    te_p = torch.from_numpy(norm.transform_patches(te_patches)).float()
    te_t = torch.from_numpy(norm.transform_tabular(te_tabular)).float()
    te_y = torch.from_numpy(te_targets).float()

    rainy_train = tr_targets[tr_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_train, 90))
    p95 = float(np.percentile(rainy_train, 95))
    thresholds = {"p90": p90, "p95": p95, "p99": float(np.percentile(rainy_train, 99))}
    print(f"  P90={p90:.1f}mm, P95={p95:.1f}mm")

    # Create P90 binary labels
    tr_p90 = (tr_targets >= p90).astype(np.float32)
    vl_p90 = (vl_targets >= p90).astype(np.float32)
    tr_p90_t = torch.from_numpy(tr_p90).float()
    vl_p90_t = torch.from_numpy(vl_p90).float()

    # Weighted sampler: oversample P90 events
    weights = np.ones(len(tr_targets))
    weights[tr_targets >= 0.1] = 2.0
    weights[tr_targets >= p90] = 10.0
    weights[tr_targets >= p95] = 20.0
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_ds = TensorDataset(tr_p, tr_t, tr_y, tr_p90_t)
    vl_ds = TensorDataset(vl_p, vl_t, vl_y, vl_p90_t)
    te_ds = TensorDataset(te_p, te_t, te_y)

    tr_loader = DataLoader(tr_ds, batch_size=64, sampler=sampler)
    vl_loader = DataLoader(vl_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

    # ── Build model with transfer learning ───────────────────────────────
    print("\n[2/5] Loading pretrained SmallNet + adding P90 head...")
    n_in = 19 * 3 * 3 + 24  # 195 features
    model = SmallNetWithP90Head(n_in)

    ckpt_path = config.OUTPUT_DIR / "window_3" / "ckpt_epoch0084_csi0.3645.pt"
    model.load_pretrained(ckpt_path)

    # Freeze backbone for first few epochs, then unfreeze
    for name, param in model.named_parameters():
        if "p90_head" not in name:
            param.requires_grad = False
    print(f"  Phase 1: Only P90 head trainable ({sum(p.numel() for p in model.parameters() if p.requires_grad)} params)")

    # ── Fine-tune ────────────────────────────────────────────────────────
    print("\n[3/5] Fine-tuning...")

    # P90 class imbalance weight
    n_pos = tr_p90.sum()
    n_neg = len(tr_p90) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_score = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, 151):
        # Phase 2: Unfreeze backbone after epoch 20
        if epoch == 21:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=130)
            print(f"\n  Phase 2: All params unfrozen ({sum(p.numel() for p in model.parameters() if p.requires_grad)} params)")

        model.train()
        epoch_loss = 0
        for batch in tr_loader:
            p_b, t_b, y_b, p90_b = batch
            optimizer.zero_grad()

            pred_reg, p90_logit = model(p_b, t_b)

            # Regression loss (Tweedie-inspired)
            mu = pred_reg.clamp(min=1e-6)
            y = y_b.clamp(min=0)
            reg_loss = F.mse_loss(mu, y)

            # P90 classification loss
            clf_loss = bce_loss_fn(p90_logit, p90_b)

            # Combined: regression + classification
            if epoch <= 20:
                loss = clf_loss  # Phase 1: only train P90 head
            else:
                loss = reg_loss + 2.0 * clf_loss  # Phase 2: both

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch <= 5:
            model.eval()
            val_preds, val_p90_probs, val_targets_list = [], [], []
            with torch.no_grad():
                for batch in vl_loader:
                    p_b, t_b, y_b, _ = batch
                    pred_reg, p90_logit = model(p_b, t_b)
                    val_preds.append(pred_reg.numpy())
                    val_p90_probs.append(torch.sigmoid(p90_logit).numpy())
                    val_targets_list.append(y_b.numpy())

            vp = np.concatenate(val_preds)
            vp90 = np.concatenate(val_p90_probs)
            vt = np.concatenate(val_targets_list)

            # Evaluate regression CSI
            m = evaluate(vp, vt, thresholds, prefix="")
            val_csi_rain = m.get("CSI_rain", 0)
            val_csi_p90 = m.get("CSI_p90", 0)

            # Also evaluate P90 classifier
            best_clf_csi = 0
            for t in np.arange(0.05, 0.9, 0.05):
                ct = contingency((vp90 >= t).astype(float) * (p90 + 1), vt, p90)
                c = csi(ct)
                if c > best_clf_csi:
                    best_clf_csi = c

            score = val_csi_rain + val_csi_p90 * 2.0 + best_clf_csi * 1.5
            if epoch % 10 == 0 or epoch <= 5:
                print(f"  E{epoch:03d} loss={epoch_loss/len(tr_loader):.4f}  "
                      f"val_CSI_rain={val_csi_rain:.4f}  val_CSI_P90={val_csi_p90:.4f}  "
                      f"val_P90_clf_CSI={best_clf_csi:.4f}  score={score:.4f}")

            if score > best_val_score:
                best_val_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 5
                if patience_counter >= 40:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # ── Also train Rain classifier (same as before) ──────────────────────
    print("\n[4/5] Training Rain/No-Rain XGBoost classifier...")
    def flatten(patches, tabulars):
        N = patches.shape[0]
        return np.hstack([patches.reshape(N, -1), tabulars])

    X_train = flatten(norm.transform_patches(tr_patches), norm.transform_tabular(tr_tabular))
    X_val = flatten(norm.transform_patches(vl_patches), norm.transform_tabular(vl_tabular))
    X_test = flatten(norm.transform_patches(te_patches), norm.transform_tabular(te_tabular))

    y_rain_tr = (tr_targets >= config.DRY_THRESHOLD).astype(int)
    y_rain_vl = (vl_targets >= config.DRY_THRESHOLD).astype(int)
    n_dry = (y_rain_tr == 0).sum()
    n_wet = (y_rain_tr == 1).sum()

    clf_rain = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42,
        scale_pos_weight=n_dry / n_wet,
    )
    clf_rain.fit(X_train, y_rain_tr, eval_set=[(X_val, y_rain_vl)], verbose=False)
    rain_prob_val = clf_rain.predict_proba(X_val)[:, 1]
    rain_prob_test = clf_rain.predict_proba(X_test)[:, 1]

    best_rain_t = 0.5
    best_rain_csi = 0
    for t in np.arange(0.2, 0.8, 0.01):
        ct = contingency((rain_prob_val >= t).astype(float) * 999, vl_targets, config.DRY_THRESHOLD)
        c = csi(ct)
        if c > best_rain_csi:
            best_rain_csi, best_rain_t = c, t
    print(f"  Rain threshold: {best_rain_t:.2f} (val CSI={best_rain_csi:.4f})")

    # ── Find best config on VALIDATION ───────────────────────────────────
    print("\n[5/5] Finding best config on VALIDATION (no leakage)...")

    # Get predictions from fine-tuned model
    with torch.no_grad():
        val_preds_all, val_p90_all = [], []
        for batch in vl_loader:
            p_b, t_b, y_b, _ = batch
            pred_reg, p90_logit = model(p_b, t_b)
            val_preds_all.append(pred_reg.numpy())
            val_p90_all.append(torch.sigmoid(p90_logit).numpy())
        val_preds = np.concatenate(val_preds_all)
        val_p90_probs = np.concatenate(val_p90_all)

        test_preds_all, test_p90_all = [], []
        for batch in te_loader:
            p_b, t_b, y_b = batch
            pred_reg, p90_logit = model(p_b, t_b)
            test_preds_all.append(pred_reg.numpy())
            test_p90_all.append(torch.sigmoid(p90_logit).numpy())
        test_preds = np.concatenate(test_preds_all)
        test_p90_probs = np.concatenate(test_p90_all)

    # Grid search on validation — try MULTIPLE combination strategies
    val_results = {}
    for rain_t in [best_rain_t - 0.05, best_rain_t, best_rain_t + 0.05]:
        for p90_t in np.arange(0.1, 0.9, 0.05):
            rain_mask = rain_prob_val >= rain_t
            p90_mask = val_p90_probs >= p90_t

            # Strategy A: No P90 boost (regression only + rain gate)
            final_a = np.zeros_like(val_preds)
            final_a[rain_mask] = val_preds[rain_mask]

            m_a = evaluate(final_a, vl_targets, thresholds, prefix="")
            score_a = (m_a.get("CSI_rain", 0) + m_a.get("CSI_p90", 0) * 2.0
                       + m_a.get("CSI_p95", 0) * 1.5 + m_a.get("SEDI_rain", 0) * 0.5)
            val_results[("noboost", rain_t, p90_t)] = {"m": m_a, "score": score_a}

            # Strategy B: Soft multiply — scales up by P90 probability
            final_b = np.zeros_like(val_preds)
            final_b[rain_mask] = val_preds[rain_mask] * (1.0 + val_p90_probs[rain_mask])

            m_b = evaluate(final_b, vl_targets, thresholds, prefix="")
            score_b = (m_b.get("CSI_rain", 0) + m_b.get("CSI_p90", 0) * 2.0
                       + m_b.get("CSI_p95", 0) * 1.5 + m_b.get("SEDI_rain", 0) * 0.5)
            val_results[("softmul", rain_t, p90_t)] = {"m": m_b, "score": score_b}

            # Strategy C: Conditional boost — only boost if regression >= p90*0.5
            final_c = np.zeros_like(val_preds)
            final_c[rain_mask] = val_preds[rain_mask]
            cond = rain_mask & p90_mask & (val_preds >= p90 * 0.5)
            final_c[cond] = np.maximum(final_c[cond], p90 * 1.02)

            m_c = evaluate(final_c, vl_targets, thresholds, prefix="")
            score_c = (m_c.get("CSI_rain", 0) + m_c.get("CSI_p90", 0) * 2.0
                       + m_c.get("CSI_p95", 0) * 1.5 + m_c.get("SEDI_rain", 0) * 0.5)
            val_results[("condboost", rain_t, p90_t)] = {"m": m_c, "score": score_c}

    best_key = max(val_results, key=lambda k: val_results[k]["score"])
    best_strat, best_rt, best_pt = best_key
    print(f"  Best: strategy={best_strat}, rain_t={best_rt:.2f}, p90_t={best_pt:.2f}")
    print(f"  Val: CSI_rain={val_results[best_key]['m'].get('CSI_rain',0):.4f}  "
          f"CSI_P90={val_results[best_key]['m'].get('CSI_p90',0):.4f}  "
          f"CSI_P95={val_results[best_key]['m'].get('CSI_p95',0):.4f}")

    # ── Apply ONCE to test ───────────────────────────────────────────────
    print("\n  Evaluating ONCE on test (no leakage)...")
    rain_mask_test = rain_prob_test >= best_rt

    final_test = np.zeros_like(test_preds)
    if best_strat == "noboost":
        final_test[rain_mask_test] = test_preds[rain_mask_test]
    elif best_strat == "softmul":
        final_test[rain_mask_test] = test_preds[rain_mask_test] * (1.0 + test_p90_probs[rain_mask_test])
    elif best_strat == "condboost":
        final_test[rain_mask_test] = test_preds[rain_mask_test]
        p90_mask_test = test_p90_probs >= best_pt
        cond = rain_mask_test & p90_mask_test & (test_preds >= p90 * 0.5)
        final_test[cond] = np.maximum(final_test[cond], p90 * 1.02)

    test_metrics = evaluate(final_test, te_targets, thresholds, prefix="")
    corr_f = np.corrcoef(final_test[te_targets >= 0.1], te_targets[te_targets >= 0.1])[0, 1] if (te_targets >= 0.1).sum() > 2 else 0

    print_metrics(test_metrics, title="TRANSFER LEARNING RESULTS (no leakage)")

    # Save
    out_dir = config.OUTPUT_DIR / "transfer_learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({
            "model": "SmallNet + P90 head (transfer learning)",
            "method": "val-optimized, single test eval (no leakage)",
            "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))},
            "thresholds": thresholds,
            "correlation": corr_f,
            "config": {"strategy": best_strat, "rain_t": best_rt, "p90_t": best_pt},
        }, f, indent=2)
    print(f"  Saved to: {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
