"""
Model Training v2
==================
Clean temporal CV on 360 country-sector-year observations.
Compares financial-only vs augmented (+ geopolitical) models.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/final_dataset_v2.csv')

FIN = ['r11_wm','r12_wm','r14_wm','r21_wm','r22_wm',
       'r25_wm','r31_wm','r32_wm','r51_wm','r52_wm','r53_wm']
GEO = ['sgpr','sgpr_lag1','gpr_global']
MAC = ['crisis_2008','brexit_period','post_covid']
ALL = FIN + GEO + MAC

LABEL = 'distress_label'
years = df['year']

# Fill missing
for col in ALL:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

print(f"Dataset: {df.shape} | Distress rate: {df[LABEL].mean()*100:.1f}%")
print(f"Years available: {sorted(df['year'].unique())}")
print()

# ── Leave-one-year-out CV ────────────────────────────────
def loyo_cv(feature_cols, label_col, model_type, model_name):
    all_true, all_prob = [], []
    test_years = [y for y in sorted(df['year'].unique())
                  if df[df['year']==y][label_col].nunique() > 1]

    for test_yr in test_years:
        train = df[df['year'] != test_yr]
        test = df[df['year'] == test_yr]

        Xtr = train[feature_cols]
        ytr = train[label_col]
        Xte = test[feature_cols]
        yte = test[label_col]

        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        if model_type == 'logistic':
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            mdl = LogisticRegression(
                class_weight='balanced', max_iter=2000,
                C=0.1, random_state=42
            )
            mdl.fit(Xtr_s, ytr)
            prob = mdl.predict_proba(Xte_s)[:,1]
        else:
            mdl = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
                eval_metric='auc', random_state=42, verbosity=0
            )
            mdl.fit(Xtr, ytr)
            prob = mdl.predict_proba(Xte)[:,1]

        all_true.extend(yte.tolist())
        all_prob.extend(prob.tolist())

    all_true = np.array(all_true)
    all_prob = np.array(all_prob)
    auc = roc_auc_score(all_true, all_prob)
    fpr, tpr, _ = roc_curve(all_true, all_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(all_true, all_prob)
    print(f"  {model_name:<50} AUC={auc:.4f}  KS={ks:.4f}  Brier={brier:.4f}")
    return auc, ks, brier, all_true, all_prob


print("="*75)
print("MODEL COMPARISON — LEAVE-ONE-YEAR-OUT CV")
print("="*75)
r1_auc,_,_,_,_ = loyo_cv(FIN,          LABEL, 'logistic', 'Logistic — Financial Only')
r2_auc,_,_,_,_ = loyo_cv(ALL,          LABEL, 'logistic', 'Logistic — Financial + Geo + Macro')
r3_auc,_,_,_,_ = loyo_cv(FIN,          LABEL, 'xgboost',  'XGBoost  — Financial Only')
r4_auc,r4_ks,r4_b,y_true,y_prob = loyo_cv(ALL, LABEL, 'xgboost', 'XGBoost  — Financial + Geo + Macro')

print()
print(f"AUC lift (Logistic):  +{(r2_auc-r1_auc)*100:.2f} pp from adding geo features")
print(f"AUC lift (XGBoost):   +{(r4_auc-r3_auc)*100:.2f} pp from adding geo features")

# ── Sector breakdown ─────────────────────────────────────
print()
print("="*75)
print("SECTOR BREAKDOWN — XGBoost Augmented")
print("="*75)
for sector in df['sector_bucket'].unique():
    m = df['sector_bucket'] == sector
    sub = df[m].copy()
    sub_years = sub['year']
    test_years = [y for y in sorted(sub['year'].unique())
                  if sub[sub['year']==y][LABEL].nunique() > 1]
    all_true, all_prob = [], []
    for test_yr in test_years:
        train = sub[sub['year'] != test_yr]
        test_s = sub[sub['year'] == test_yr]
        Xtr, ytr = train[ALL], train[LABEL]
        Xte, yte = test_s[ALL], test_s[LABEL]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        mdl = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
            eval_metric='auc', random_state=42, verbosity=0
        )
        mdl.fit(Xtr.fillna(Xtr.median()), ytr)
        prob = mdl.predict_proba(Xte.fillna(Xtr.median()))[:,1]
        all_true.extend(yte.tolist())
        all_prob.extend(prob.tolist())
    if len(set(all_true)) > 1:
        auc = roc_auc_score(all_true, all_prob)
        n = len(sub)
        dr = sub[LABEL].mean()*100
        print(f"  {sector:<20} AUC={auc:.4f}  n={n}  distress={dr:.1f}%")

# ── Country breakdown ────────────────────────────────────
print()
print("="*75)
print("COUNTRY BREAKDOWN — XGBoost Augmented")
print("="*75)
for country in sorted(df['country'].unique()):
    m = df['country'] == country
    sub = df[m].copy()
    test_years = [y for y in sorted(sub['year'].unique())
                  if sub[sub['year']==y][LABEL].nunique() > 1]
    all_true, all_prob = [], []
    for test_yr in test_years:
        train = sub[sub['year'] != test_yr]
        test_s = sub[sub['year'] == test_yr]
        Xtr, ytr = train[ALL], train[LABEL]
        Xte, yte = test_s[ALL], test_s[LABEL]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(train) < 5:
            continue
        mdl = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
            eval_metric='auc', random_state=42, verbosity=0
        )
        mdl.fit(Xtr.fillna(Xtr.median()), ytr)
        prob = mdl.predict_proba(Xte.fillna(Xtr.median()))[:,1]
        all_true.extend(yte.tolist())
        all_prob.extend(prob.tolist())
    if len(set(all_true)) > 1:
        auc = roc_auc_score(all_true, all_prob)
        print(f"  {country:<6} AUC={auc:.4f}  n={m.sum()}")