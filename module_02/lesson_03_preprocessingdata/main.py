import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df = pd.read_csv(url, header = None)  
df.columns = headers

#Modifies de dataframe - if inplace == False, generates a new dataframe
df.replace('?',np.NaN, inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)

mean = df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean, inplace=True)
