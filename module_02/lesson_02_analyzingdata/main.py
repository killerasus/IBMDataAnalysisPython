import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df = pd.read_csv(url, header = None)  
df.columns = headers

print(df.head(5))

df1 = df.replace('?',np.NaN)
df=df1.dropna(subset=["price"], axis=0)
print(df.head(20))

print(df.dtypes)

print(df.describe())
print(df.describe(include="all")) #includes object type data
print(df[['length','compression-ratio']].describe())

print(df.info())

df.to_csv("./automobile.csv", index=False)