import pandas as pd

from utils.all_utils import X_y_split, save_plot
from utils.model import Perceptron

def main(df, modelName, plotName, eta, epochs):
    X,y = X_y_split(df)
    print(f"X:  {X.shape}\ny:  {y.shape}")

    model_or = Perceptron(eta=eta, epochs=epochs)
    model_or.fit(X, y)

    _ = model_or.total_loss()

    model_or.save(file_name=modelName)

    save_plot(df=df, model=model_or, filename=plotName)


if __name__ == "__main__":
    OR = pd.DataFrame({"x1":(0,0,1,1), 
                       "x2":(0,1,0,1), 
                       "y":(0,1,1,1)})

    ETA = 0.1
    EPOCHS = 10

    main(df=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
