import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("full_hi_en.csv")

print("Median cosine similarity:", df["cosine_before_after"].median())
nice_T = df["cosine_before_after"].quantile(0.95)

beautiful = df[df["cosine_before_after"] >= nice_T]
print(beautiful.sample(20)[["tweet", "en_text", "cosine_before_after"]])
