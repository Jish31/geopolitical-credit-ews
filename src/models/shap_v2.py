"""
SHAP Analysis v2
=================
Trains final XGBoost augmented model on full dataset
and generates SHAP explainability outputs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from xgboost import XGBClassifier

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/processed/final_dataset_v2.csv')

FIN = ['r11_wm','r12_wm','r14_wm','r21_wm','r22_wm',
       'r25_wm','r31_wm','r32_wm','r51_wm','r52_wm','r53_wm']
GEO = ['sgpr','sgpr_lag1','gpr_global']
MAC = ['crisis_2008','brexit_period','post_covid']
ALL = FIN + GEO + MAC
LABEL = 'distress_label'

X = df[ALL].fillna(df[ALL].median())
y = df[LABEL]

# Train final model on full dataset
print("Training final XGBoost augmented model...")
model = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    eval_metric='auc', random_state=42, verbosity=0
)
model.fit(X, y)
joblib.dump(model, 'outputs/xgb_final_v2.pkl')
print("Model saved: outputs/xgb_final_v2.pkl")

# SHAP values
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_df = pd.DataFrame(shap_values, columns=ALL)
shap_df.to_csv('outputs/shap_values_v2.csv', index=False)

# ── Feature group attribution ─────────────────────────────
geo_shap  = shap_df[GEO].abs().sum().sum()
fin_shap  = shap_df[FIN].abs().sum().sum()
mac_shap  = shap_df[MAC].abs().sum().sum()
total     = geo_shap + fin_shap + mac_shap

print()
print("="*50)
print("SHAP ATTRIBUTION BY FEATURE GROUP")
print("="*50)
print(f"  Geopolitical: {geo_shap/total*100:.1f}%")
print(f"  Financial:    {fin_shap/total*100:.1f}%")
print(f"  Macro:        {mac_shap/total*100:.1f}%")

print()
print("Feature importance by mean |SHAP|:")
mean_shap = shap_df.abs().mean().sort_values(ascending=False)
for feat, val in mean_shap.items():
    group = 'GEO' if feat in GEO else 'FIN' if feat in FIN else 'MAC'
    print(f"  [{group}] {feat:<20} {val:.4f}")

# ── Beeswarm plot ────────────────────────────────────────
print()
print("Generating SHAP plots...")
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X, feature_names=ALL,
                  show=False, max_display=18)
plt.title("SHAP Feature Importance — XGBoost Augmented (v2)")
plt.tight_layout()
plt.savefig('outputs/shap_beeswarm_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/shap_beeswarm_v2.png")

# ── Bar plot ─────────────────────────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X, feature_names=ALL,
                  plot_type='bar', show=False, max_display=18)
plt.title("Mean |SHAP| — Feature Importance Ranking (v2)")
plt.tight_layout()
plt.savefig('outputs/shap_bar_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/shap_bar_v2.png")

# ── Sector SHAP comparison ───────────────────────────────
print()
print("="*50)
print("SHAP GEO CONTRIBUTION BY SECTOR")
print("="*50)
for sector in df['sector_bucket'].unique():
    m = df['sector_bucket'] == sector
    geo_s = shap_df[m][GEO].abs().sum().sum()
    fin_s = shap_df[m][FIN].abs().sum().sum()
    tot_s = geo_s + fin_s + shap_df[m][MAC].abs().sum().sum()
    print(f"  {sector:<20} Geo={geo_s/tot_s*100:.1f}%  Fin={fin_s/tot_s*100:.1f}%")

print()
print("All outputs saved to outputs/")