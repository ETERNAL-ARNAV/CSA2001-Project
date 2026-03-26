import os
from src.predict import returnTOMain
if(returnTOMain() == True):
    # 1. Clean the data
    os.system("python src/preprocess.py")

    # 2. Train the brain
    os.system("python src/train_model.py")
elif(returnTOMain() == False):
    pass
else:
    # 3. Start the predictor
    os.system("python src/predict.py")