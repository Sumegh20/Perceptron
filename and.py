import os
import pandas as pd

from utils.all_utils import X_y_split, save_plot
from utils.model import Perceptron
import logging

gate = 'OR Gate'
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "logfile.log"),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
    )

def main(df, modelName, plotName, eta, epochs):
    X,y = X_y_split(df)
    print(f"X:  {X.shape}\ny:  {y.shape}")

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(file_name=modelName)

    save_plot(df=df, model=model, filename=plotName)


if __name__ == "__main__":
    AND = pd.DataFrame({"x1":(0,0,1,1), 
                        "x2":(0,1,0,1), 
                        "y":(0,0,0,1)})

    ETA = 0.1
    EPOCHS = 10

    try:
        logging.info(f"_____START TRANING_____  {gate}")
        main(df=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(f'_____DONE TRANING_____  {gate}\n\n')
    except Exception as e:
        logging.exception(e)
        raise e
