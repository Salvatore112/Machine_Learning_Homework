from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
import numpy as np
import csv as csv

os.chdir(r"./Data/")

data = pd.read_csv("train.csv")
predict = pd.read_csv("test.csv")



target = "life_expectancy"

data.ffill(inplace=True)
predict.ffill(inplace=True)
for i in range(0, len(data["country"].values)):
    data["country"].values[i] = hash(data["country"].values[i])

for j in range(0, len(data["status"].values)):
    data["status"].values[j] = hash(data["status"].values[j])


for i in range(0, len(predict["country"].values)):
    predict["country"].values[i] = hash(predict["country"].values[i])

for j in range(0, len(predict["status"].values)):
    predict["status"].values[j] = hash(predict["status"].values[j])

features = data.columns.tolist()
features.remove(target)

model = RandomForestRegressor(n_jobs=-1)
model.fit(data[features].values, data[target].values)

y_new = model.predict(predict[features].values)
rows = []
rows.append(["id", "life_expectancy"])

for i in range(0, len(y_new)):
    rows.append([i, y_new[i]])
print(rows)

a = np.asarray(rows)

outfile = open('./fin_ids.csv','w', newline="")
out = csv.writer(outfile)
out.writerows(rows)
outfile.close()
