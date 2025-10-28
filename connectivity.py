import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

conn_df = pd.read_csv('conn_prob.csv', index_col=0)
mask = conn_df == 0
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Connection Probability'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 Laminar Connectivity Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()



conn_df = pd.read_csv('conductances.csv', index_col=0)
mask = conn_df == 0
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Conductance values'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 Laminar Conductance scaled realistic Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()


plt.show()
