# Projet Visualisation et Exploration des Donnees

## Ce que demande le sujet

Le projet suit un pipeline complet de data science :

1. Comprendre les donnees
2. Visualiser les variables et leurs relations
3. Transformer les variables
4. Entrainer un modele de classification
5. Predire quels clients contacter
6. Presenter les resultats dans un dashboard Streamlit
7. Rediger un court rapport de synthese

## Fichiers disponibles

- `Data/train_info.csv` : jeu d'apprentissage avec la cible `reponse_client`
- `Data/clients_a_contacter.csv` : jeu de prediction sans la cible
- `Sujet du projet.pdf` : enonce

## Plan step by step

### 1. Charger et inspecter les donnees

- Charger `train_info.csv` et `clients_a_contacter.csv`
- Verifier les dimensions
- Verifier les types de variables
- Identifier :
  - l'identifiant : `id_client`
  - la cible : `reponse_client`
  - les variables quantitatives
  - les variables categorielles
  - les variables binaires codees en nombres
- Verifier :
  - valeurs manquantes
  - doublons
  - valeurs aberrantes

### 2. Faire l'EDA

- Resume statistique des variables quantitatives
- Distribution de la cible `reponse_client`
- Graphiques pour les variables categorielles :
  - `genre`
  - `age_vehicule`
  - `vehicule_endommage`
  - `ancien_assure`
  - `permis_conduire`
- Graphiques pour les variables quantitatives :
  - histogrammes
  - boxplots
- Analyse des relations avec la cible :
  - taux de reponse par modalite
  - comparaison des distributions selon la cible
- Correlation de Spearman entre variables numeriques

### 3. Preparer les variables

- Convertir les petites cardinalites en type `category`
- Creer une variable `tranche_age`
- Creer quelques interactions utiles, par exemple :
  - `tranche_age` x `age_vehicule`
  - `vehicule_endommage` x `ancien_assure`
- Encoder :
  - One-hot pour `genre`, `vehicule_endommage`, `tranche_age`
  - ordinal pour `age_vehicule`
- Traiter `code_regional` et `canal_communication` comme variables a forte cardinalite
- Standardiser uniquement apres le split train/test

### 4. Entrainer et evaluer un modele

- Separarer `X` et `y`
- Faire un `train_test_split` stratifie
- Tester au moins 2 modeles, par exemple :
  - `RandomForestClassifier`
  - `XGBClassifier` si disponible
  - sinon `HistGradientBoostingClassifier`
- Evaluer avec des metriques adaptees au desequilibre :
  - precision
  - recall
  - f1-score
  - ROC-AUC
- Comparer les modeles et garder le meilleur

### 5. Predire les clients a contacter

- Reutiliser exactement les memes transformations que pour le train
- Predire la probabilite de `reponse_client = 1`
- Classer les clients par probabilite
- Construire une liste cible :
  - eviter les clients presque certains de souscrire
  - eviter les clients tres peu susceptibles de souscrire
  - privilegier la zone intermediaire si l'objectif est d'optimiser l'action commerciale

### 6. Produire les resultats metier

- Exporter un CSV avec :
  - `id_client`
  - `proba_reponse`
  - `classe_predite`
  - `segment_contact`
- Expliquer quels profils sont les plus interessants a contacter
- Ajouter quelques chiffres de synthese :
  - nombre de clients cibles
  - prime moyenne
  - age moyen
  - repartition par genre / age vehicule / dommage

### 7. Construire le dashboard Streamlit

L'application peut contenir ces onglets :

- `Apercu`
- `EDA`
- `Cible`
- `Modelisation`
- `Clients a contacter`
- `Qualite des donnees`

### 8. Rediger le rapport

Le sujet attend un rapport synthetique de 2 a 4 pages :

- contexte
- donnees
- resultats d'EDA
- transformations retenues
- modele choisi
- resultats et recommandations metier

## Ordre de travail conseille

1. Finir l'analyse exploratoire dans un notebook
2. Construire un pipeline de preprocessing
3. Tester plusieurs modeles
4. Generer les predictions sur `clients_a_contacter.csv`
5. Construire le dashboard Streamlit
6. Rediger le rapport final

## Livrables attendus

- un notebook d'analyse
- une application Streamlit fonctionnelle
- un rapport de synthese
- un fichier de prediction / ciblage client
