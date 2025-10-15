import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the matrix
conn_df = pd.read_csv('scaled_matrix_0_to_0.34.csv', index_col=0)
mask = conn_df == 0
# Visualize
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Connection Probability'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 Laminar Connectivity Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()

# Load the matrix
conn_df = pd.read_csv('conductance.csv', index_col=0)
mask = conn_df == 0
# Visualize
plt.figure(figsize=(7, 7))
sns.heatmap(conn_df, annot=True, fmt='.3f', cmap='Reds', square=True,
            linewidths=0.3, cbar_kws={'label': 'Conductance values'}, mask=mask, annot_kws={'size': 6}
            )

plt.title('V1 Laminar Conductance Matrix')
plt.xlabel('Postsynaptic Population')
plt.ylabel('Presynaptic Population')
plt.tight_layout()
plt.show()