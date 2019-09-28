import kmean_create_data as kcd
import kmean_shawn_simister as kss
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data = {"x":[], "y":[], "cluster":[]}

for i in range(len(kss.assignment_values)):
    data["x"].append(kcd.vectors_set[i][0])
    data["y"].append(kcd.vectors_set[i][1])
    data["cluster"].append(kss.assignment_values[i])
    
df = pd.DataFrame(data)
sns.lmplot("x", "y", data = df, fit_reg = False, size = 6, hue = "cluster", legend = False)
plt.show()