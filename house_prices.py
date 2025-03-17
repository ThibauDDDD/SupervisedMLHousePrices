import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)  # permet d'afficher le print en entier

def get_data():
    """get the data from the csv files

    Returns:
        _type_: the differents data
    """
    #je récupère les données des csv
    csv_train = pd.read_csv("train.csv")
    csv_true_test = pd.read_csv("test.csv")

    #je trie pour ne récupérer que les données qui m'intéressent
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    target = "SalePrice"
    csv_train_sorted_x = csv_train[features]
    csv_train_y = csv_train[target]

    #je divise le csv en 2 parties, une partie train avec 80/100 des données, et une partie test avec 20/100 des données
    limit = int(0.8 * len(csv_train_sorted_x))
    csv_test_sorted_x = csv_train_sorted_x[limit:]
    csv_test_sorted_y = csv_train_y[limit:]
    csv_train_sorted_x = csv_train_sorted_x[:limit]
    csv_train_y = csv_train_y[:limit]

    #je transforme les csv en np array, et je transforme les tableaux 2d Y en tableau 1d pour que l'algo fonctionne
    train_x = csv_train_sorted_x.to_numpy()
    train_y = csv_train_y.to_numpy().ravel().astype(int)
    test_x = csv_test_sorted_x.to_numpy()
    test_y = csv_test_sorted_y.to_numpy().ravel().astype(int)

    #je normalise les données (ça permet d'avoir des données avec le même impact sur l'algo)
    scaled_train_x = StandardScaler().fit_transform(train_x)
    scaled_test_x = StandardScaler().fit_transform(test_x)
    return (scaled_train_x, scaled_test_x, train_x, test_x, train_y, test_y)


def get_true_test():
    """get the data from the test.csv file, for the kaggle competition

    Returns:
        _type_: the differents data
    """
    #je récupère les données des csv
    csv_true_test = pd.read_csv("test.csv")

    #je trie pour ne récupérer que les données qui m'intéressent
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    id = csv_true_test["Id"]
    csv_true_test = csv_true_test[features]

    #je transforme les csv en np array, et je transforme les tableaux 2d Y en tableau 1d pour que l'algo fonctionne
    true_test = csv_true_test.to_numpy()

    #je normalise les données (ça permet d'avoir des données avec le même impact sur l'algo)
    scaled_true_test = StandardScaler().fit_transform(true_test)
    return (true_test, scaled_true_test, id)


def print_graph_LinearRegression(test_y, y_pred, accuracy: float): 
    """create a graph for the prediction of the house prices with linear regression

    Args:
        test_y (ndarray): the Real Price of the house
        y_pred (ndarray): The prediction of the price of the house
        accuracy (float): the accuracy of the prediction
    """
    plt.figure()
    plt.scatter(x=test_y, y=y_pred, marker='x')
    plt.plot([min(test_y), max(test_y)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label="Idéal")
    plt.xlabel("Prix Réel, Précision de la prédiction: " + str(int(accuracy * 100)) + "%")
    plt.ylabel("Prix Prédit")
    plt.title("Prédiction des prix des maisons")
    plt.grid(True)
    plt.savefig("Prédiction_prix_maison_LinearRegression.png", dpi=300)

def print_graph_RandomForestRegressor(test_y, y_pred, accuracy: float): 
    """create a graph for the prediction of the house prices with Random Forest Regressor

    Args:
        test_y (ndarray): the Real Price of the house
        y_pred (ndarray): The prediction of the price of the house
        accuracy (float): the accuracy of the prediction
    """
    plt.figure()
    plt.scatter(x=test_y, y=y_pred, marker='x')
    plt.plot([min(test_y), max(test_y)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label="Idéal")
    plt.xlabel("Prix Réel, Précision de la prédiction: " + str(int(accuracy * 100)) + "%")
    plt.ylabel("Prix Prédit")
    plt.title("Prédiction des prix des maisons")
    plt.grid(True)
    plt.savefig("Prédiction_prix_maison_RandomForestRegressor.png", dpi=300)

def main():
    #je récupère les données
    scaled_train_x, scaled_test_x, train_x, test_x, train_y, test_y = get_data()
    true_test, scaled_true_test, id = get_true_test()

    #je crée l'algo linear regression, je l'entraîne et je le teste. Enfin, je fais un graphique avec les résultats
    algoLinearRegression = LinearRegression().fit(scaled_train_x, train_y)
    y_predLinearRegression = algoLinearRegression.predict(scaled_test_x)
    accuracyLinearRegression = algoLinearRegression.score(scaled_test_x, test_y)
    print_graph_LinearRegression(test_y, y_predLinearRegression, accuracyLinearRegression)

    #je crée l'algo Random Forest Regressor, je l'entraîne
    algoRandomForestRegressor = RandomForestRegressor().fit(scaled_train_x, train_y)

    # cette partie est là pour tester sur le fichier test.csv et rendre au format csv le résultat de ce test. 
    y_pred_true_test = algoRandomForestRegressor.predict(scaled_true_test)
    sample = pd.DataFrame(y_pred_true_test, columns=["SalePrice"])
    sample = pd.concat([id, sample], axis=1)
    sample.to_csv("submission.csv", index=False)

    # Je teste et je fais un graphique avec les résultats.
    y_predRandomForestRegressor = algoRandomForestRegressor.predict(scaled_test_x)
    accuracyRandomForestRegressor = algoRandomForestRegressor.score(scaled_test_x, test_y)
    print_graph_RandomForestRegressor(test_y, y_predRandomForestRegressor, accuracyRandomForestRegressor)

if __name__ == "__main__":
    main()