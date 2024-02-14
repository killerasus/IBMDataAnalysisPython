import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df = pd.read_csv(url, header = None)  
df.columns = headers

print(df["city-mpg"])

# Converts mpg to L/100km in Car dataset
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg" : "city-L/100km"}, inplace=True)

print(df["city-L/100km"])

df.replace('?',np.NaN, inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)
# If whe check df.dtypes, we're going to seet price as object
# Converts price from object to int
df["price"] = df["price"].astype("int")

# Binning

bins = np.linspace(min(df["price"]), max(df["price"]), 4)
group_names = ["Low", "Medium", "High"]

df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)

print(df[["price", "price-binned"]])