import pandas
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import os

data = pandas.read_csv("Python/Grade predictor/data/studentData.csv")
X = data[["hours_studied", "sleep_hours"]]
y = data["score"]

model = LinearRegression()
model.fit(X, y)

baseDir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(baseDir, "model.pkl")
joblib.dump(model, modelPath)
print(f"Model trained and saved to {modelPath}")

plt.scatter(data["hours_studied"], y)
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study vs Score")

graphPath = os.path.join(baseDir, "graph.png")
plt.savefig(graphPath)