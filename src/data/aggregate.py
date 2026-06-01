import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/final_modelling_dataset.csv')

FIN = ['r11_wm','r12_wm','r14_wm','r21_wm','r22_wm',
       'r25_wm','r31_wm','r32_wm','r51_wm','r52_wm','r53_wm']

agg = df.groupby(['country','sector_bucket','year']).agg(
    **{f: (f, 'mean') for f in FIN},
    sgpr=('sgpr','first'),
    sgpr_lag1=('sgpr_lag1','first'),
    gpr_global=('gpr_global','first'),
    gdp_growth_pct=('gdp_growth_pct','first'),
    crisis_period=('crisis_period','first'),
    post_covid=('post_covid','first'),
    brexit_period=('brexit_period','first'),
    distress_label=('distress_label','first')
).reset_index()

print('Aggregated dataset shape:', agg.shape)
print('Distress rate:', round(agg['distress_label'].mean()*100, 1), '%')
print('Years:', agg['year'].min(), '-', agg['year'].max())
print('Rows:', len(agg))
print()
print('Distress by year:')
print(pd.crosstab(agg['year'], agg['distress_label']))

agg.to_csv('data/processed/aggregated_dataset.csv', index=False)
print()
print('Saved: data/processed/aggregated_dataset.csv')