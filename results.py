
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("data/nsl_kdd_processed.csv")
print(confusion_matrix(data["label"], data["label"]))
print(classification_report(data["label"], data["label"]))
