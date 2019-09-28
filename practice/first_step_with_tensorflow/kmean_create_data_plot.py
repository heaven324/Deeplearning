# 데이터 시각화
import kmean_create_data as kcd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in kcd.vectors_set], "y": [v[1] for v in kcd.vectors_set]})
sns.lmplot("x", "y", data = df, fit_reg = False, size = 6)
plt.show()i