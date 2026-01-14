import pandas as pd

train = pd.read_csv('master_train.csv', engine='python')
test = pd.read_csv('master_test.csv', engine='python')

print(f"master_train.csv: {train['tweet'].nunique()} unique tweets (from {len(train)} rows)")
print(f"master_test.csv: {test['tweet'].nunique()} unique tweets (from {len(test)} rows)")
