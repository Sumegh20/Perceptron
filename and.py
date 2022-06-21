import pandas as pd

from utils.all_utils import X_y_split, save_plot
from utils.model import Perceptron

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

    main(df=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
