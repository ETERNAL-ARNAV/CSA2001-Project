import os
from src.predict import returnTOMain
from art import *
print(text2art("---\t\t\t\t\t Spam Detector ---"))
if(returnTOMain()):
    # 1. Clean the data
    os.system("python src/preprocess.py")

    # 2. Train the brain
    os.system("python src/train_model.py")
else:
    # 3. Start the predictor
    os.system("python src/predict.py")