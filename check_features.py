import pandas as pd

data = pd.read_csv(r"F:\Sepsis_Early_Prediction_12hr\data\processed\cleaned_12hr_data.csv", nrows=0)

print(list(data.columns))
