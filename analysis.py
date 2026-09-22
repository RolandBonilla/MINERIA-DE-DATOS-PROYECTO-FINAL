import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

plt.rcParams.update({'font.size': 10})

df = pd.read_csv('data/telecom_churn_dataset.csv')

report = {}
report['n_rows_raw'] = len(df)
report['n_cols'] = df.shape[1]

# ---------- EDA ----------
nulls_pct = (df.isna().sum() / len(df) * 100).round(2)
report['nulls_pct'] = nulls_pct[nulls_pct > 0].to_dict()

dup_count = df.duplicated(subset=[c for c in df.columns if c != 'customerID']).sum()
report['dup_count'] = int(dup_count)
report['dup_pct'] = round(dup_count / len(df) * 100, 2)

# TotalCharges numeric coercion for stats
df['TotalCharges_num'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

desc = df[['tenure', 'MonthlyCharges', 'TotalCharges_num']].describe().round(2)
report['descriptive_stats'] = desc.to_dict()

# Outliers via IQR on MonthlyCharges
q1, q3 = df['MonthlyCharges'].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = df[(df['MonthlyCharges'] < low) | (df['MonthlyCharges'] > high)]
report['outliers_monthlycharges'] = {
    'q1': round(q1,2), 'q3': round(q3,2), 'iqr': round(iqr,2),
    'lower_bound': round(low,2), 'upper_bound': round(high,2),
    'n_outliers': len(outliers), 'pct_outliers': round(len(outliers)/len(df)*100,2)
}

# ---------- Visualizations ----------
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['tenure'], bins=20, color='#3b6ea5', edgecolor='white')
ax.set_title('Distribución de la antigüedad del cliente (tenure)')
ax.set_xlabel('Meses de antigüedad'); ax.set_ylabel('Frecuencia')
plt.tight_layout(); plt.savefig('figs/fig1_hist_tenure.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6,4))
ax.boxplot(df['MonthlyCharges'].dropna(), vert=True, patch_artist=True,
           boxprops=dict(facecolor='#a5c8e1'))
ax.set_title('Diagrama de cajas: cargos mensuales (MonthlyCharges)')
ax.set_ylabel('USD')
plt.tight_layout(); plt.savefig('figs/fig2_box_monthlycharges.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6,4))
ct = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
ct.plot(kind='bar', ax=ax, color=['#4c8bd6','#d65f4c'])
ax.set_title('Porcentaje de fuga (Churn) por tipo de contrato')
ax.set_ylabel('% de clientes'); ax.set_xlabel('Tipo de contrato')
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig('figs/fig3_bar_churn_contract.png', dpi=150); plt.close()

# ---------- Preprocessing ----------
work = df.drop(columns=['customerID', 'TotalCharges']).copy()
work = work.rename(columns={'TotalCharges_num':'TotalCharges'})

work.drop_duplicates(inplace=True)
report['n_rows_after_dedup'] = len(work)

work['TotalCharges'] = work['TotalCharges'].fillna(work['TotalCharges'].median())

# Feature engineering: 2 derived variables
work['Cargo_Promedio_Mensual'] = np.where(
    work['tenure'] > 0, work['TotalCharges'] / work['tenure'], work['MonthlyCharges']
)
n_services = work[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies']].apply(lambda c: (c=='Yes').astype(int)).sum(axis=1)
work['Indice_Fidelidad'] = (n_services * (work['tenure'] + 1)) / (work['MonthlyCharges'] + 1)

# cap outliers (IQR winsorizing) on MonthlyCharges
work['MonthlyCharges'] = work['MonthlyCharges'].clip(low, high)

cat_cols = work.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Churn')
num_cols = [c for c in work.columns if c not in cat_cols + ['Churn']]

report['feature_engineering'] = {
    'Cargo_Promedio_Mensual': 'TotalCharges / tenure (o MonthlyCharges si tenure=0)',
    'Indice_Fidelidad': '(N servicios contratados x (tenure+1)) / (MonthlyCharges+1)'
}

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, RocCurveDisplay)

X = work.drop(columns=['Churn'])
y = (work['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
report['split'] = {'train': len(X_train), 'test': len(X_test), 'train_pct': 80, 'test_pct': 20}

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

models = {
    'Regresión Logística': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(max_depth=6, criterion='gini', min_samples_leaf=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, criterion='gini', random_state=42)
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]

    cv = cross_validate(pipe, X_train, y_train, cv=skf,
                         scoring=['accuracy','precision','recall','f1','roc_auc'])

    results[name] = {
        'accuracy': round(accuracy_score(y_test, y_pred),4),
        'precision': round(precision_score(y_test, y_pred),4),
        'recall': round(recall_score(y_test, y_pred),4),
        'f1': round(f1_score(y_test, y_pred),4),
        'auc_roc': round(roc_auc_score(y_test, y_proba),4),
        'cv_accuracy_mean': round(cv['test_accuracy'].mean(),4),
        'cv_accuracy_std': round(cv['test_accuracy'].std(),4),
        'cv_f1_mean': round(cv['test_f1'].mean(),4),
        'cv_f1_std': round(cv['test_f1'].std(),4),
        'cv_auc_mean': round(cv['test_roc_auc'].mean(),4),
        'cv_auc_std': round(cv['test_roc_auc'].std(),4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

report['model_results'] = results
report['hyperparameters'] = {
    'Regresión Logística': {'solver':'lbfgs','C':1.0,'max_iter':1000},
    'Árbol de Decisión': {'max_depth':6,'criterion':'gini','min_samples_leaf':20},
    'Random Forest': {'n_estimators':200,'max_depth':8,'criterion':'gini'}
}

# ROC curves figure
fig, ax = plt.subplots(figsize=(6,5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.set_title('Curvas ROC comparativas de los modelos')
plt.tight_layout(); plt.savefig('figs/fig4_roc_curves.png', dpi=150); plt.close()

# Confusion matrix best model (highest F1)
best_name = max(results, key=lambda k: results[k]['f1'])
cm = np.array(results[best_name]['confusion_matrix'])
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['No Fuga','Fuga']); ax.set_yticklabels(['No Fuga','Fuga'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center', color='black', fontsize=12)
ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
ax.set_title(f'Matriz de confusión — {best_name}')
plt.tight_layout(); plt.savefig('figs/fig5_confusion_matrix.png', dpi=150); plt.close()

# Feature importance (Random Forest)
rf_pipe = fitted['Random Forest']
ohe = rf_pipe.named_steps['pre'].named_transformers_['cat']
feat_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
importances = rf_pipe.named_steps['clf'].feature_importances_
imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(6,4.5))
imp_series.sort_values().plot(kind='barh', ax=ax, color='#4c8bd6')
ax.set_title('Importancia de variables — Random Forest (top 10)')
plt.tight_layout(); plt.savefig('figs/fig6_feature_importance.png', dpi=150); plt.close()

report['best_model'] = best_name
report['top_features'] = imp_series.round(4).to_dict()

with open('report_data.json','w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=== RESUMEN ===")
print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])

# ---------- GENERAR MODELO SERIALIZADO ----------
import joblib
joblib.dump(fitted['Regresión Logística'], 'modelo_churn.pkl')
print("¡El archivo modelo_churn.pkl se ha creado exitosamente en tu carpeta!")