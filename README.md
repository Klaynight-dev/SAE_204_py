# 📊 Analyse de Corrélation - Longueur des Noms et Performances Académiques

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*Une analyse statistique pour explorer la relation entre la longueur des noms d'étudiants et leurs performances académiques*

</div>

---

## 🎯 **Objectif du Projet**

Ce projet analyse s'il existe une corrélation significative entre la longueur combinée des noms et prénoms des étudiants et leurs moyennes académiques. L'analyse comprend des calculs statistiques manuels, des visualisations interactives et des tests de significativité.

## 📋 **Fonctionnalités**

### ✨ **Analyses Statistiques**
- 📈 **Corrélation de Pearson** (calcul manuel)
- 📊 **Corrélation de Spearman** (calcul manuel)
- 🧮 **Statistiques descriptives** complètes
- 🎲 **Tests de significativité**

### 📊 **Visualisations**
- 🔍 **Nuages de points** avec lignes de tendance
- 📈 **Histogrammes** de distribution
- 🌡️ **Heatmap** des corrélations
- 📦 **Box plots** par groupes

### 💾 **Exports**
- 📄 **Matrice de corrélation** (CSV)
- 🖼️ **Graphiques haute résolution** (PNG)
- 📝 **Rapport statistique** détaillé

---

## 🚀 **Installation Rapide**

### **Prérequis**
```bash
Python 3.8+
```

### **Installation des dépendances**
```bash
pip install -r requirements.txt
```

### **Structure du projet**
```
📁 SAE/
├── 📄 auter.py                    # Script principal
├── 📄 VueSAE.csv                  # Données étudiants
├── 📄 requirements.txt           # Dépendances
├── 📄 README.md                  # Documentation
└── 📁 outputs/                   # Résultats générés
    ├── 🖼️ analyse_correlation.png
    ├── 🖼️ analyse_semestres.png
    └── 📄 matrice_correlation.csv
```

---

## 🎮 **Utilisation**

### **Exécution Simple**
```bash
python auter.py
```

### **Données d'entrée**
Le fichier `VueSAE.csv` doit contenir :
- `nom` : Nom de famille
- `prenom` : Prénom
- `moyennesemestre1` : Moyenne du 1er semestre
- `moyennesemestre2` : Moyenne du 2ème semestre

---

## 📊 **Résultats d'Analyse**

### **Statistiques Calculées**

| Métrique | Description |
|----------|-------------|
| **Corrélation de Pearson** | Relation linéaire entre variables |
| **Corrélation de Spearman** | Relation monotone (non-paramétrique) |
| **Coefficient R²** | Pourcentage de variance expliquée |
| **Test t de Student** | Significativité statistique |
| **P-value** | Probabilité d'erreur de type I |

### **Interprétation des Corrélations**

```
📊 Force de la corrélation :
   • |r| < 0.1  → Très faible
   • |r| < 0.3  → Faible  
   • |r| < 0.5  → Modérée
   • |r| < 0.7  → Forte
   • |r| ≥ 0.7  → Très forte

🎯 Significativité :
   • p < 0.05  → Statistiquement significative
   • p ≥ 0.05  → Non significative
```

---

## 📈 **Exemples de Sortie**

### **Console Output**
```
=============================================================
ANALYSE DE CORRÉLATION
=============================================================

=== STATISTIQUES COMPLÈTES (CALCUL MANUEL) ===
Longueur Nom+Prénom vs Moyenne Générale:
Nombre d'observations: 1048

Moyennes:
  • Longueur moyenne: 18.42
  • Moyenne générale: 11.87

Corrélations:
  • Pearson: -0.0156
  • Spearman: -0.0203
  • R² (coefficient de détermination): 0.0002

Interprétation: Corrélation très faible
La corrélation n'est PAS statistiquement significative (p >= 0.05)
```

### **Visualisations Générées**

<table>
<tr>
<td width="50%">

**📊 Analyse Principale**
- Nuage de points avec tendance
- Distribution des longueurs
- Distribution des moyennes  
- Heatmap des corrélations

</td>
<td width="50%">

**📈 Analyse par Semestre**
- Corrélations individuelles
- Comparaison inter-semestres
- Box plots par groupes
- Analyse descriptive

</td>
</tr>
</table>

---

## 🛠️ **Configuration Avancée**

### **Personnalisation des Seuils**
```python
# Modifier les groupes de longueur
q1 = donnees['longueur_nom_prenom'].quantile(0.25)  # 1er quartile
q3 = donnees['longueur_nom_prenom'].quantile(0.75)  # 3ème quartile
```

### **Export Personnalisé**
```python
# Sauvegarder avec format spécifique
plt.savefig('mon_analyse.pdf', format='pdf', dpi=300)
matrice_corr.to_excel('resultats.xlsx')
```

---

## 🔧 **Dépendances**

| Package | Version | Usage |
|---------|---------|-------|
| `pandas` | ≥1.5.0 | Manipulation des données |
| `numpy` | ≥1.21.0 | Calculs numériques |
| `matplotlib` | ≥3.5.0 | Graphiques de base |
| `seaborn` | ≥0.11.0 | Visualisations avancées |
| `scipy` | ≥1.8.0 | Tests statistiques |

---

## 📝 **Notes Importantes**

### ⚠️ **Limitations**
- Les calculs de p-value sont approximatifs
- Les données manquantes sont automatiquement exclues
- L'analyse suppose une distribution normale pour certains tests

### 🔍 **Recommandations**
- Vérifiez la qualité des données avant analyse
- Interprétez les résultats dans le contexte académique
- Considérez d'autres facteurs influençant les performances

---

## 📞 **Support**

### **Problèmes Courants**

<details>
<summary>🔴 Erreur de chargement CSV</summary>

```bash
# Vérifiez l'encodage du fichier
df = pd.read_csv('VueSAE.csv', encoding='utf-8-sig', sep=';')
```
</details>

<details>
<summary>🟡 Valeurs manquantes</summary>

```python
# Le script gère automatiquement les NaN
# Vérifiez vos données avec :
print(df.isnull().sum())
```
</details>

<details>
<summary>🟢 Performances lentes</summary>

```python
# Pour de gros datasets, utilisez :
df = df.sample(n=1000)  # Échantillonnage
```
</details>

---

## 📄 **License**

```
MIT License - Libre d'utilisation pour l'éducation et la recherche
```

---

<div align="center">

**Made with ❤️ by <a href="https://github.com/Klaynight-dev/">Klaynight</a>**

[⬆ Retour en haut](#-analyse-de-corrélation---longueur-des-noms-et-performances-académiques)

</div>
