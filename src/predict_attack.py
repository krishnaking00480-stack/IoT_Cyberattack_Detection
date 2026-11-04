
import pandas as pd
import joblib

# Load trained AI model
model = joblib.load("models/cyberattack_model.pkl")

# Dataset files
benign_path = "dataset/BenignTraffic1.pcap.csv"
attack_path = "dataset/DDoS-UDP_Flood.pcap.csv"

# Load sample traffic records
benign = pd.read_csv(benign_path, nrows=1)
attack = pd.read_csv(attack_path, nrows=1)

# Prediction function
def predict_traffic(data, actual_type):

    prediction = model.predict(data)[0]

    if prediction == 0:
        result = "Normal Traffic"
    else:
        result = "UDP Flood Attack"

    print("\nActual Traffic:", actual_type)
    print("AI Prediction:", result)

    if prediction == 1:
        print("ALERT: Suspicious traffic detected!")
    else:
        print("Status: Traffic appears normal.")

# Predict normal traffic
predict_traffic(benign, "Normal Traffic")

# Predict attack traffic
predict_traffic(attack, "UDP Flood Attack")
