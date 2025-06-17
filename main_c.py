import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def analyser_correlation():
    """
    Analyse la corrélation entre la longueur des noms+prénoms et les moyennes des étudiants
    """
    
    # Charger les données
    try:
        df = pd.read_csv('VueSAE.csv', encoding='utf-8', sep=';')
    except:
        try:
            df = pd.read_csv('VueSAE.csv', encoding='latin-1', sep=';')
        except:
            df = pd.read_csv('VueSAE.csv', encoding='utf-8', sep=',')
    
    print("Données chargées avec succès!")
    print(f"Nombre d'étudiants: {len(df)}")
    print("\nColonnes disponibles:")
    print(df.columns.tolist())
    
    # Calculer la longueur des noms + prénoms
    df['longueur_nom_prenom'] = (df['nom'].astype(str) + df['prenom'].astype(str)).str.len()
    
    # Identifier les colonnes de moyennes - version pour vos nouvelles données
    colonnes_moyennes = []
    
    # Chercher spécifiquement les colonnes moyennesemestre1 et moyennesemestre2
    if 'moyennesemestre1' in df.columns:
        colonnes_moyennes.append('moyennesemestre1')
    if 'moyennesemestre2' in df.columns:
        colonnes_moyennes.append('moyennesemestre2')
    
    # Si les noms sont différents, chercher automatiquement
    if not colonnes_moyennes:
        for col in df.columns:
            if 'moyenne' in col.lower() and 'semestre' in col.lower():
                colonnes_moyennes.append(col)
    
    print(f"\nColonnes de moyennes identifiées: {colonnes_moyennes}")
    
    if len(colonnes_moyennes) == 0:
        print("Aucune colonne de moyennes trouvée!")
        return None, None
    
    # Afficher quelques exemples de valeurs pour vérification
    print(f"\nExemples de valeurs pour chaque colonne:")
    for col in colonnes_moyennes:
        valeurs_non_nulles = df[col].dropna()
        if len(valeurs_non_nulles) > 0:
            print(f"  {col}: min={valeurs_non_nulles.min():.2f}, max={valeurs_non_nulles.max():.2f}, moy={valeurs_non_nulles.mean():.2f}")
        else:
            print(f"  {col}: Aucune valeur valide")
    
    # Calculer la moyenne générale des deux semestres
    if len(colonnes_moyennes) >= 2:
        df['moyenne_generale'] = df[colonnes_moyennes].mean(axis=1, skipna=True)
        print(f"\nMoyenne générale calculée à partir de {len(colonnes_moyennes)} semestres")
    else:
        # Si on n'a qu'un seul semestre, l'utiliser comme moyenne générale
        df['moyenne_generale'] = df[colonnes_moyennes[0]]
        print(f"\nMoyenne générale = {colonnes_moyennes[0]} (un seul semestre trouvé)")
    
    # Préparer les données pour l'analyse de corrélation
    donnees_correlation = df[['longueur_nom_prenom', 'moyenne_generale'] + colonnes_moyennes].copy()
    
    # Supprimer les lignes avec des valeurs manquantes dans longueur_nom_prenom et moyenne_generale
    donnees_correlation = donnees_correlation.dropna(subset=['longueur_nom_prenom', 'moyenne_generale'])
    
    print(f"\nNombre d'observations pour l'analyse: {len(donnees_correlation)}")
    
    if len(donnees_correlation) == 0:
        print("Aucune donnée valide pour l'analyse!")
        return None, None
    
    # === ANALYSE DE CORRÉLATION ===
    print("\n" + "="*60)
    print("ANALYSE DE CORRÉLATION")
    print("="*60)
    
    # Corrélation entre longueur nom+prénom et moyenne générale
    def calculer_statistiques_manuelles(x, y):
        """
        Calcule toutes les statistiques de corrélation manuellement sans fonctions externes
        
        Args:
            x: série de données (longueur nom+prénom)
            y: série de données (moyennes)
        
        Returns:
            dict: dictionnaire contenant toutes les statistiques calculées
        """
        n = len(x)
        
        # Moyennes
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calcul des sommes pour variances et covariance
        sum_xx = sum((xi - mean_x) ** 2 for xi in x)
        sum_yy = sum((yi - mean_y) ** 2 for yi in y)
        sum_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        # Variances (avec correction de Bessel n-1)
        var_x = sum_xx / (n - 1)
        var_y = sum_yy / (n - 1)
        
        # Écarts-types
        std_x = (var_x) ** 0.5
        std_y = (var_y) ** 0.5
        
        # Covariance
        covariance = sum_xy / (n - 1)
        
        # Coefficient de corrélation de Pearson
        corr_pearson = sum_xy / ((sum_xx * sum_yy) ** 0.5)
        
        # Coefficient de détermination (R²)
        r_squared = corr_pearson ** 2
        
        # Test de significativité (statistique t)
        if n > 2:
            t_stat = corr_pearson * ((n - 2) / (1 - r_squared)) ** 0.5
            # Approximation simple de la p-value (non exacte mais indicative)
            p_value_approx = 2 * (1 - abs(t_stat) / (abs(t_stat) + (n - 2) ** 0.5))
        else:
            t_stat = 0
            p_value_approx = 1
        
        # Corrélation de Spearman (calcul manuel des rangs)
        def calculer_rangs(data):
            # Trier les valeurs avec leurs indices originaux
            valeurs_triees = sorted(enumerate(data), key=lambda x: x[1])
            rangs = [0] * len(data)
            
            # Assigner les rangs
            for i, (index_original, valeur) in enumerate(valeurs_triees):
                rangs[index_original] = i + 1
            
            # Gérer les égalités (rang moyen)
            i = 0
            while i < len(valeurs_triees):
                j = i
                while j < len(valeurs_triees) - 1 and valeurs_triees[j][1] == valeurs_triees[j + 1][1]:
                    j += 1
                
                if i != j:  # Il y a des égalités
                    rang_moyen = (i + j + 2) / 2  # +2 car les rangs commencent à 1
                    for k in range(i, j + 1):
                        index_original = valeurs_triees[k][0]
                        rangs[index_original] = rang_moyen
                
                i = j + 1
            
            return rangs
        
        rangs_x = calculer_rangs(x)
        rangs_y = calculer_rangs(y)
        
        # Calculer la corrélation de Spearman sur les rangs
        mean_rang_x = sum(rangs_x) / n
        mean_rang_y = sum(rangs_y) / n
        
        sum_rang_xx = sum((rx - mean_rang_x) ** 2 for rx in rangs_x)
        sum_rang_yy = sum((ry - mean_rang_y) ** 2 for ry in rangs_y)
        sum_rang_xy = sum((rx - mean_rang_x) * (ry - mean_rang_y) for rx, ry in zip(rangs_x, rangs_y))
        
        if sum_rang_xx > 0 and sum_rang_yy > 0:
            corr_spearman = sum_rang_xy / ((sum_rang_xx * sum_rang_yy) ** 0.5)
        else:
            corr_spearman = 0
        
        return {
            'n': n,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'var_x': var_x,
            'var_y': var_y,
            'std_x': std_x,
            'std_y': std_y,
            'covariance': covariance,
            'pearson': corr_pearson,
            'pearson_t': t_stat,
            'pearson_p_approx': p_value_approx,
            'spearman': corr_spearman,
            'r_squared': r_squared
        }

    # Utilisation de la fonction
    x_data = donnees_correlation['longueur_nom_prenom'].tolist()
    y_data = donnees_correlation['moyenne_generale'].tolist()
    
    stats = calculer_statistiques_manuelles(x_data, y_data)

    print(f"\n=== STATISTIQUES COMPLÈTES (CALCUL MANUEL) ===")
    print(f"Longueur Nom+Prénom vs Moyenne Générale:")
    print(f"Nombre d'observations: {stats['n']}")
    
    print(f"\nMoyennes:")
    print(f"  • Longueur moyenne: {stats['mean_x']:.2f}")
    print(f"  • Moyenne générale: {stats['mean_y']:.2f}")
    
    print(f"\nVariances:")
    print(f"  • Variance longueur: {stats['var_x']:.4f}")
    print(f"  • Variance moyenne: {stats['var_y']:.4f}")
    
    print(f"\nÉcarts-types:")
    print(f"  • Écart-type longueur: {stats['std_x']:.4f}")
    print(f"  • Écart-type moyenne: {stats['std_y']:.4f}")
    
    print(f"\nCovariance: {stats['covariance']:.6f}")
    
    print(f"\nCorrélations:")
    print(f"  • Pearson: {stats['pearson']:.6f}")
    print(f"  • Spearman: {stats['spearman']:.6f}")
    print(f"  • R² (coefficient de détermination): {stats['r_squared']:.6f}")
    
    print(f"\nTest de significativité (Pearson):")
    print(f"  • Statistique t: {stats['pearson_t']:.4f}")
    print(f"  • P-value approximative: {stats['pearson_p_approx']:.6f}")
    
    # Variables pour la suite du code
    corr_pearson = stats['pearson']
    p_value_pearson = stats['pearson_p_approx']
    # Interprétation
    def interpreter_correlation(corr):
        if abs(corr) < 0.1:
            return "très faible"
        elif abs(corr) < 0.3:
            return "faible"
        elif abs(corr) < 0.5:
            return "modérée"
        elif abs(corr) < 0.7:
            return "forte"
        else:
            return "très forte"
    
    print(f"\nInterprétation: Corrélation {interpreter_correlation(corr_pearson)}")
    if p_value_pearson < 0.05:
        print("La corrélation est statistiquement significative (p < 0.05)")
    else:
        print("La corrélation n'est PAS statistiquement significative (p >= 0.05)")
    
    # Corrélations avec chaque semestre
    print(f"\nCorrélations avec chaque semestre:")
    print("-" * 50)
    for semestre in colonnes_moyennes:
        # Supprimer les valeurs manquantes pour ce semestre
        donnees_temp = donnees_correlation[['longueur_nom_prenom', semestre]].dropna()
        if len(donnees_temp) > 1:
            corr, p_val = pearsonr(donnees_temp['longueur_nom_prenom'], 
                                 donnees_temp[semestre])
            print(f"  {semestre:20}: {corr:7.4f} (p={p_val:.4f}) - {len(donnees_temp)} obs.")
    
    # === MATRICE DE CORRÉLATION ===
    print("\n" + "="*60)
    print("MATRICE DE CORRÉLATION")
    print("="*60)
    
    # Calculer la matrice de corrélation (en gérant les valeurs manquantes)
    matrice_corr = donnees_correlation[['longueur_nom_prenom', 'moyenne_generale'] + colonnes_moyennes].corr()
    
    print("\nMatrice de corrélation complète:")
    print(matrice_corr.round(4))
    
    # Sauvegarder la matrice dans un fichier CSV
    matrice_corr.to_csv('matrice_correlation.csv')
    print("\nMatrice sauvegardée dans 'matrice_correlation.csv'")
    
    # === VISUALISATIONS ===
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Nuage de points principal
    axes[0,0].scatter(donnees_correlation['longueur_nom_prenom'], 
                     donnees_correlation['moyenne_generale'], 
                     alpha=0.6, color='blue')
    axes[0,0].set_xlabel('Longueur Nom + Prénom')
    axes[0,0].set_ylabel('Moyenne Générale (2 semestres)')
    axes[0,0].set_title(f'Corrélation: {corr_pearson:.3f}')
    
    # Ligne de tendance
    z = np.polyfit(donnees_correlation['longueur_nom_prenom'], 
                   donnees_correlation['moyenne_generale'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(donnees_correlation['longueur_nom_prenom'], 
                   p(donnees_correlation['longueur_nom_prenom']), 
                   "r--", alpha=0.8)
    
    # 2. Histogramme des longueurs
    axes[0,1].hist(donnees_correlation['longueur_nom_prenom'], bins=20, alpha=0.7, color='green')
    axes[0,1].set_xlabel('Longueur Nom + Prénom')
    axes[0,1].set_ylabel('Fréquence')
    axes[0,1].set_title('Distribution des longueurs')
    
    # 3. Histogramme des moyennes
    axes[1,0].hist(donnees_correlation['moyenne_generale'], bins=20, alpha=0.7, color='orange')
    axes[1,0].set_xlabel('Moyenne Générale')
    axes[1,0].set_ylabel('Fréquence')
    axes[1,0].set_title('Distribution des moyennes')
    
    # 4. Heatmap de la matrice de corrélation
    # Utiliser seaborn pour une meilleure heatmap
    sns.heatmap(matrice_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1,1], fmt='.3f', cbar_kws={'shrink': 0.8})
    axes[1,1].set_title('Matrice de Corrélation')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig('analyse_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === GRAPHIQUE SUPPLÉMENTAIRE: COMPARAISON DES SEMESTRES ===
    if len(colonnes_moyennes) >= 2:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(donnees_correlation['longueur_nom_prenom'], 
                   donnees_correlation[colonnes_moyennes[0]], 
                   alpha=0.6, color='red', label=colonnes_moyennes[0])
        plt.xlabel('Longueur Nom + Prénom')
        plt.ylabel(colonnes_moyennes[0])
        plt.title(f'Corrélation avec {colonnes_moyennes[0]}')
        
        plt.subplot(2, 2, 2)
        plt.scatter(donnees_correlation['longueur_nom_prenom'], 
                   donnees_correlation[colonnes_moyennes[1]], 
                   alpha=0.6, color='blue', label=colonnes_moyennes[1])
        plt.xlabel('Longueur Nom + Prénom')
        plt.ylabel(colonnes_moyennes[1])
        plt.title(f'Corrélation avec {colonnes_moyennes[1]}')
        
        plt.subplot(2, 2, 3)
        plt.scatter(donnees_correlation[colonnes_moyennes[0]], 
                   donnees_correlation[colonnes_moyennes[1]], 
                   alpha=0.6, color='purple')
        plt.xlabel(colonnes_moyennes[0])
        plt.ylabel(colonnes_moyennes[1])
        plt.title('Corrélation entre les 2 semestres')
        
        plt.subplot(2, 2, 4)
        # Boxplot par groupe de longueur
        q1 = donnees_correlation['longueur_nom_prenom'].quantile(0.33)
        q2 = donnees_correlation['longueur_nom_prenom'].quantile(0.67)
        
        donnees_correlation['groupe_longueur'] = pd.cut(
            donnees_correlation['longueur_nom_prenom'], 
            bins=[0, q1, q2, float('inf')], 
            labels=['Court', 'Moyen', 'Long']
        )
        
        donnees_correlation.boxplot(column='moyenne_generale', by='groupe_longueur', ax=plt.gca())
        plt.title('Moyennes par groupe de longueur')
        plt.suptitle('')  # Supprimer le titre automatique de pandas
        
        plt.tight_layout()
        plt.savefig('analyse_semestres.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # === STATISTIQUES DESCRIPTIVES ===
    print("\n" + "="*60)
    print("STATISTIQUES DESCRIPTIVES")
    print("="*60)
    
    print(f"\nLongueur Nom+Prénom:")
    print(f"  Moyenne: {donnees_correlation['longueur_nom_prenom'].mean():.2f}")
    print(f"  Médiane: {donnees_correlation['longueur_nom_prenom'].median():.2f}")
    print(f"  Min-Max: {donnees_correlation['longueur_nom_prenom'].min()}-{donnees_correlation['longueur_nom_prenom'].max()}")
    print(f"  Écart-type: {donnees_correlation['longueur_nom_prenom'].std():.2f}")
    
    print(f"\nMoyenne Générale:")
    print(f"  Moyenne: {donnees_correlation['moyenne_generale'].mean():.2f}")
    print(f"  Médiane: {donnees_correlation['moyenne_generale'].median():.2f}")
    print(f"  Min-Max: {donnees_correlation['moyenne_generale'].min():.2f}-{donnees_correlation['moyenne_generale'].max():.2f}")
    print(f"  Écart-type: {donnees_correlation['moyenne_generale'].std():.2f}")
    
    # Statistiques par semestre
    for semestre in colonnes_moyennes:
        print(f"\n{semestre}:")
        semestre_data = donnees_correlation[semestre].dropna()
        if len(semestre_data) > 0:
            print(f"  Moyenne: {semestre_data.mean():.2f}")
            print(f"  Médiane: {semestre_data.median():.2f}")
            print(f"  Min-Max: {semestre_data.min():.2f}-{semestre_data.max():.2f}")
            print(f"  Écart-type: {semestre_data.std():.2f}")
    
    # === ANALYSE PAR GROUPES ===
    print("\n" + "="*60)
    print("ANALYSE PAR GROUPES DE LONGUEUR")
    print("="*60)
    
    # Créer des groupes basés sur la longueur
    q1 = donnees_correlation['longueur_nom_prenom'].quantile(0.33)
    q2 = donnees_correlation['longueur_nom_prenom'].quantile(0.67)
    
    donnees_correlation['groupe_longueur'] = pd.cut(
        donnees_correlation['longueur_nom_prenom'], 
        bins=[0, q1, q2, float('inf')], 
        labels=['Court', 'Moyen', 'Long']
    )
    
    groupes = donnees_correlation.groupby('groupe_longueur')[['moyenne_generale'] + colonnes_moyennes].agg(['count', 'mean', 'std'])
    print("\nStatistiques par groupe de longueur:")
    print(groupes.round(3))
    
    return donnees_correlation, matrice_corr

if __name__ == "__main__":
    print("Démarrage de l'analyse de corrélation...")
    result = analyser_correlation()
    if result[0] is not None:
        donnees, matrice = result
        print("\nAnalyse terminée!")
    else:
        print("\nAnalyse interrompue à cause d'erreurs dans les données.")