import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


conn_df = pd.read_csv('config/connection_probabilities2.csv', index_col=0)
mask = conn_df == 0
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Connection Probability2'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 Connectivity Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()



conn_df = pd.read_csv('config/conductances_AMPA2.csv', index_col=0)
mask = conn_df == 0
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Conductance values'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 conductances_AMPA  Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()

conn_df = pd.read_csv('config/conductances_NMDA2.csv', index_col=0)
mask = conn_df == 0
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Conductance values'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 conductances_NMDA Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()


plt.show()
