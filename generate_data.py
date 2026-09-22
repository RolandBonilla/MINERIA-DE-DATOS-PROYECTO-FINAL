import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

gender = rng.choice(['Male','Female'], n)
senior = rng.choice([0,1], n, p=[0.84,0.16])
partner = rng.choice(['Yes','No'], n, p=[0.48,0.52])
dependents = rng.choice(['Yes','No'], n, p=[0.30,0.70])
tenure = rng.integers(0,73,n)
phone = rng.choice(['Yes','No'], n, p=[0.90,0.10])
multiplelines = np.where(phone=='No','No phone service', rng.choice(['Yes','No'], n, p=[0.42,0.58]))
internet = rng.choice(['DSL','Fiber optic','No'], n, p=[0.34,0.44,0.22])
def dep_service(internet_arr):
    out=[]
    for x in internet_arr:
        if x=='No': out.append('No internet service')
        else: out.append(rng.choice(['Yes','No'], p=[0.29,0.71]))
    return np.array(out)
onlinesecurity = dep_service(internet)
onlinebackup = dep_service(internet)
deviceprotection = dep_service(internet)
techsupport = dep_service(internet)
streamingtv = dep_service(internet)
streamingmovies = dep_service(internet)
contract = rng.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.24,0.21])
paperless = rng.choice(['Yes','No'], n, p=[0.59,0.41])
payment = rng.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], n, p=[0.34,0.23,0.22,0.21])

base_charge = np.where(internet=='Fiber optic', rng.normal(85,12,n),
              np.where(internet=='DSL', rng.normal(58,10,n), rng.normal(21,5,n)))
addon_cost = (
    (onlinesecurity=='Yes').astype(int)*5 + (onlinebackup=='Yes').astype(int)*5 +
    (deviceprotection=='Yes').astype(int)*5 + (techsupport=='Yes').astype(int)*5 +
    (streamingtv=='Yes').astype(int)*7 + (streamingmovies=='Yes').astype(int)*7
)
monthlycharges = np.clip(base_charge + addon_cost + rng.normal(0,3,n), 18, 120).round(2)
totalcharges = np.clip(monthlycharges * tenure + rng.normal(0,20,n), 0, None).round(2)

# introduce some missing/duplicate/outlier noise deliberately (realistic messiness)
totalcharges_str = totalcharges.astype(object)
miss_idx = rng.choice(n, size=int(n*0.012), replace=False)
for i in miss_idx: totalcharges_str[i] = np.nan

# churn probability model (logistic-ish) based on real-world drivers
logit = (
    -1.2
    + 1.6*(contract=='Month-to-month')
    - 1.1*(contract=='Two year')
    - 0.35*(contract=='One year')
    - 0.03*tenure
    + 0.018*monthlycharges
    + 0.5*(internet=='Fiber optic')
    - 0.4*(onlinesecurity=='Yes')
    - 0.4*(techsupport=='Yes')
    + 0.3*(paperless=='Yes')
    + 0.4*(payment=='Electronic check')
    - 0.25*(partner=='Yes')
    - 0.2*(dependents=='Yes')
    + rng.normal(0,0.6,n)
)
prob = 1/(1+np.exp(-logit))
churn = (rng.uniform(0,1,n) < prob).astype(int)
churn_label = np.where(churn==1,'Yes','No')

df = pd.DataFrame({
    'customerID':[f'C{10000+i}' for i in range(n)],
    'gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':dependents,
    'tenure':tenure,'PhoneService':phone,'MultipleLines':multiplelines,'InternetService':internet,
    'OnlineSecurity':onlinesecurity,'OnlineBackup':onlinebackup,'DeviceProtection':deviceprotection,
    'TechSupport':techsupport,'StreamingTV':streamingtv,'StreamingMovies':streamingmovies,
    'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,
    'MonthlyCharges':monthlycharges,'TotalCharges':totalcharges_str,'Churn':churn_label
})

# duplicate some rows
dup_idx = rng.choice(n, size=int(n*0.018), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# a few extreme outliers in MonthlyCharges
out_idx = rng.choice(len(df), size=6, replace=False)
df.loc[out_idx,'MonthlyCharges'] = df.loc[out_idx,'MonthlyCharges'] + rng.uniform(150,220,6)

df.to_csv('data/telecom_churn_dataset.csv', index=False)
print(df.shape)
print(df.head(3).to_string())
