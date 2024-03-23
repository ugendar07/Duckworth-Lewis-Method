import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None]*10
        self.L = None
        self.total_loss = None

        # self.Z0,self.L = Params
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        if Z_0 is None:
            Z_0 = self.Z0[w-1]
        if L is None:
            L = self.L
        # print("in pred ",self.Z0)
        # print(X.shape)
        pred = Z_0 * (1 - np.exp(-(L * X)/ Z_0))
        # print("pred vlaue :",pred)
        return pred

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """

        total_loss = 0
        # print("y values:",Y,"\n",Y.shape,"ef",len(Y))
        # w = data['Wickets.in.Hand'].values
        # print("\n wickets in hand ")
        # print(Params[9])
        # print(w[63])
        for i in range(0,len(Y)):
            loss=(Y[i]-self.get_predictions(X[i],Z_0=Params[w[i]-1],L=Params[10]))**2
            total_loss+=loss
        # print("model total loss",total_loss)
        return total_loss
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.
return pd.read_csv('../Assi_1/04_cricket_1999to2011.csv')

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv('../data/04_cricket_1999to2011.csv')
    return df



def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    cleaned_data = data.copy()
    # a = len(cleaned_data)
    # print(a)
    cleaned_data = cleaned_data[cleaned_data['Error.In.Data'] == 0]  # Remove rows with Error.In.Data values = 1
    # print(len(cleaned_data))
    # print(a-len(cleaned_data))
    cleaned_data = pd.DataFrame(data,columns = ['Match', 'Innings', 'Overs.Remaining','Runs.Remaining' ,'Wickets.in.Hand','Over', 'Innings.Total.Runs'])
    inning_first = cleaned_data['Innings'] == 1
    cleaned_data = cleaned_data[inning_first]
    # print(len(cleaned_data))
    cleaned_data['Overs.Remaining'] = 50-cleaned_data['Over']
    # cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], format='%d/%m/%Y')
    cleaned_data = cleaned_data.dropna()  # Remove rows with missing values
    # print(len(cleaned_data))

    return cleaned_data




def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    
    X = data['Overs.Remaining'].values
    Y = data['Runs.Remaining'].values
    W = data["Wickets.in.Hand"].values
    Z_0 = []
    for wickets in range(1,11):
        temp= data[data['Wickets.in.Hand'] == wickets]
        Z_0.append( np.mean(temp['Runs.Remaining'].values))
    l=1
    Z_0.append(l)

    # print(Z_0)
    optimized_params = sp.optimize.minimize(
        model.calculate_loss,Z_0, args=(X, Y , W),method='L-BFGS-B')
   
    # print("optimized_params",optimized_params.x)
    model.Z0 =optimized_params.x
    # print(optimized_params)
    model.L = optimized_params.x[10]
    model.total_loss = optimized_params.fun
    # print("fun value ",optimized_params.fun)
    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    # print("\n in plot")
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    plt.xlabel('Overs Remaining')
    plt.ylabel('Average Runs Scored')
    plt.title('Run Production Functions for each Wicket')

    overs = np.arange(51)
    # linear_overs = 50.0 - overs
    # slope = -4.89 * overs + 244
    # print(model.Z0[10-1])
    for i in range(0, 10):
        y = model.get_predictions(overs, Z_0=model.Z0[i], L=model.Z0[10])
        plt.plot(overs, y, label='Z0 for '+str(i+1)+' wickets = '+str("{:.00f}".format(model.Z0[i])))
        
    # plt.plot(linear_overs, slope, color='black', label='Average Run Rate')
    plt.legend()
    plt.savefig(plot_path)
    plt.show()

    

def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    print("\n")
    print("*"*60)
    print("\nModel Parameters: ")
    for i in range(10):
        print('Z_0(' + str(i + 1) + ') = ' + str("{:.0f}".format(model.Z0[i])))
    print("\nL:", model.L)
    print("\n")
    print("*"*60)
    print("\nMinimized Total Loss :", model.total_loss)
    return model.Z0 + [model.L]


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    # print("in normal calculate_loss")
    X = data['Overs.Remaining'].values
    Y = data['Runs.Remaining'].values
    W = data["Wickets.in.Hand"].values
    # print(len(Y))
    # print(Y.shape)

    total_loss = 0.0
    for i in range(len(Y)):
        loss = (Y[i] - model.get_predictions(X[i], Z_0=model.Z0[W[i]-1], L=model.L)) ** 2
        total_loss += loss
    # print("total loss",total_loss)
    normalized_loss = total_loss / len(Y)
    # print("norm loss",normalized_loss)
    return normalized_loss


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    # print(data.columns)
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    # print(data.columns)
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    print("\nThe Normalized Loss :",calculate_loss(model, data))
    # print(calculate_loss(model, data))


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
