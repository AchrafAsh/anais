import pandas as pd
import matplotlib.pyplot as plt

filename = "clean_dataset.csv"

df = pd.read_csv(filename)
print(df.columns)

df["code"] = df["code"].apply(lambda code: code.replace(" ", ""))

output_vocab = df["code"].unique()
print(len(output_vocab))  # 226

occurences = df.groupby(["code"])["input"].count()
print(f"max occurence: {occurences.max()} | min occurence: {occurences.min()}")

# 159 (70% of our dataset only appears once)
print(
    f"Occurences (once): {occurences[occurences == 1].count() / len(df.code.unique()):2.2%}")
# 211 (93% appears three times or less)
print(
    f"Occurences (3 or less): {occurences[occurences <= 3].count() / len(df.code.unique()):2.2%}")

print(occurences[occurences == 12])
print(df[df["code"] == "NLRTM"])

# plt.boxplot(occurences)
plt.show()
