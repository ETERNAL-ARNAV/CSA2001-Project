import os
from src.predict import returnTOMain
print("--- 🚀 STARTING THE SPAM FILTER PIPELINE ---")
if(returnTOMain()):
    # 1. Clean the data
    os.system("python src/preprocess.py")

    # 2. Train the brain
    os.system("python src/train_model.py")

# 3. Start the predictor
os.system("python src/predict.py")