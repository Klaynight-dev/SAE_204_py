# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:59:20 2025

@author: Tiphaine Jezequel,
         Elouan Passereau,
         Titouan Gernion,
         Ji-o Kim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients de la régression linéaire multiple.
    Args:
        X (array-like): Matrice des caractéristiques (n_samples, n_features).
        y (array-like): Vecteur des valeurs cibles (n_samples,).
    Returns:
        array: Coefficients de la régression linéaire (n_features + 1
        pour l'ordonnée à l'origine).
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta.flatten() 

def coefficient_correlation_multiple(y_true, y_pred):
    """
    Calcule le coefficient de détermination R² pour la régression linéaire.
    Args:
        y_true (array-like): Valeurs réelles.
        y_pred (array-like): Valeurs prédites par le modèle.
    Returns:
        float: Coefficient de détermination R².
    """
    y_true = np.ravel(y_true) 
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot 

# Traitement CSV
data = pd.read_csv('VueSAE.csv', sep=';')
data = data.dropna()

# Calcul de la longueur du nom+prénom (supposant des colonnes 'nom' et 'prenom')
# Adaptez les noms de colonnes selon votre fichier CSV
data['longueur_nom_prenom'] = (data['nom'].astype(str) + data['prenom'].astype(str)).str.len()

# Sélection des colonnes pour l'analyse de corrélation
colonnes_analyse = ['longueur_nom_prenom', 'moyennesemestre1', 'moyennesemestre2']
# Vérifier quelles colonnes existent réellement dans le DataFrame
colonnes_existantes = [col for col in colonnes_analyse if col in data.columns]
print(f"Colonnes disponibles: {list(data.columns)}")
print(f"Colonnes utilisées pour l'analyse: {colonnes_existantes}")
data_correlation = data[colonnes_existantes].select_dtypes(include=[np.number])

# Préparation pour la régression (longueur comme variable explicative)
X = data_correlation[['longueur_nom_prenom']].values.astype(float)
y_moyenne1 = data_correlation['moyennesemestre1'].values.astype(float)
y_moyenne2 = data_correlation['moyennesemestre2'].values.astype(float)

# Matrice de corrélation
correlation_matrix = data_correlation.corr()

# Graphiques
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Matrice de corrélation (heatmap)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Matrice de corrélation\n(Longueur nom+prénom vs Notes)')

# 2. Scatter plot longueur vs moyenne1
# 2. Scatter plot longueur vs moyenne1
ax2.scatter(data_correlation['longueur_nom_prenom'], data_correlation['moyennesemestre1'], alpha=0.7)
ax2.set_xlabel('Longueur nom+prénom')
ax2.set_ylabel('Moyenne Semestre 1')
ax2.set_title('Longueur nom+prénom vs Moyenne Semestre 1')
ax2.grid(True)

# 3. Scatter plot longueur vs moyenne2
ax3.scatter(data_correlation['longueur_nom_prenom'], data_correlation['moyennesemestre2'], alpha=0.7)
ax3.set_xlabel('Longueur nom+prénom')
ax3.set_ylabel('Moyenne Semestre 2')
ax3.set_title('Longueur nom+prénom vs Moyenne Semestre 2')
ax3.grid(True)
# 4. Diagramme en barres des corrélations avec la longueur nom+prénom
correlations_longueur = correlation_matrix['longueur_nom_prenom'][1:]  # Exclure la corrélation avec elle-même
ax4.bar(range(len(correlations_longueur)), correlations_longueur.values)
ax4.set_xlabel('Variables de notes')
ax4.set_ylabel('Corrélation avec longueur nom+prénom')
ax4.set_title('Corrélations avec longueur nom+prénom')
ax4.set_xticks(range(len(correlations_longueur)))
ax4.set_xticklabels(correlations_longueur.index, rotation=45, ha='right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Affichage des coefficients de corrélation
# Affichage des coefficients de corrélation
print("Coefficients de corrélation avec la longueur nom+prénom:")
for col in ['moyennesemestre1', 'moyennesemestre2']:
    corr = correlation_matrix.loc['longueur_nom_prenom', col]
    print(f"- {col}: {corr:.4f}")
