import pandas as pd
import os
import random

# --------------------------------------------
# File path configuration
# --------------------------------------------
dataset_path = os.path.join("data", "processed", "cleaned_12hr_data.csv")

if not os.path.exists(dataset_path):
    print("❌ Dataset not found! Make sure 'cleaned_12hr_data.csv' is in the 'data' folder.")
    exit()

# --------------------------------------------
# Load dataset
# --------------------------------------------
data = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded successfully with {len(data)} rows and {len(data.columns)} columns.")

# --------------------------------------------
# Verify key columns
# --------------------------------------------
if "Patient_ID" not in data.columns or "SepsisLabel" not in data.columns:
    print("❌ Dataset missing required columns: 'Patient_ID' and/or 'SepsisLabel'")
    exit()

# --------------------------------------------
# Get unique patient-level sepsis status
# --------------------------------------------
# Each patient considered sepsis-positive if any of their rows has SepsisLabel == 1
patient_labels = data.groupby("Patient_ID")["SepsisLabel"].max().reset_index()

# Separate lists of positive and negative patients
sepsis_positive_ids = patient_labels[patient_labels["SepsisLabel"] == 1]["Patient_ID"].tolist()
non_sepsis_ids = patient_labels[patient_labels["SepsisLabel"] == 0]["Patient_ID"].tolist()

print(f"🧍 Total patients: {len(patient_labels)}")
print(f"🟥 Sepsis-positive patients: {len(sepsis_positive_ids)}")
print(f"🟩 Non-sepsis patients: {len(non_sepsis_ids)}")

# --------------------------------------------
# Choose patient type
# --------------------------------------------
print("\nChoose which patient to extract:")
print("1️⃣  Sepsis-positive patient (has SepsisLabel = 1)")
print("2️⃣  Non-sepsis patient (never had SepsisLabel = 1)")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    selected_id = random.choice(sepsis_positive_ids)
    category = "sepsis_positive"
elif choice == "2":
    selected_id = random.choice(non_sepsis_ids)
    category = "non_sepsis"
else:
    print("❌ Invalid choice. Please enter 1 or 2.")
    exit()

# --------------------------------------------
# Extract that patient's data
# --------------------------------------------
patient_data = data[data["Patient_ID"] == selected_id]

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Save as a new CSV file
output_file = os.path.join("data", f"single_patient_{category}_{selected_id}.csv")
patient_data.to_csv(output_file, index=False)

print(f"\n✅ Saved patient file for ID={selected_id} ({category}) with {len(patient_data)} rows.")
print(f"📂 File saved at: {output_file}")
