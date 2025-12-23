import joblib

model = joblib.load("models/random_forest_model.pkl")
print(model.feature_names_in_)
