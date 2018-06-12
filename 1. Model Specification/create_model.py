import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values # our input features
Y = training_data_df[['total_earnings']].values # our output features/what we are trying to predict

# Define the model
