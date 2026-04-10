import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
# On utilise train_info pour l'exploration car il contient les réponses clients
df = pd.read_csv('train_info.csv')

# 2. Résumé statistique et typologie
print("--- Dimensions du dataset ---")
print(df.shape)
print("\n--- Aperçu des données ---")
display(df.head())
print("\n--- Statistiques descriptives ---")
display(df.describe())
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

# 3. Visualisation des variables catégorielles (ex: Genre)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='genre', hue='reponse_client')
plt.title('Distribution du Genre par Réponse Client')
plt.show()

# 4. Visualisation des variables quantitatives (ex: Âge)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='reponse_client', kde=True, element="step")
plt.title('Distribution de l\'âge selon la souscription')
plt.show()

# 5. Analyse de corrélation (Spearman)
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr(method='spearman')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation de Spearman')
plt.show()