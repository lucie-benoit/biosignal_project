# Détection de la Fibrillation Auriculaire par ECG

Projet de réimplémentation d'un pipeline de traitement du signal et de machine learning pour la détection automatique de fibrillation auriculaire (FA) à partir d'enregistrements ECG courts, dans le cadre du module de Traitement du Signal.

## Contexte

La fibrillation auriculaire (FA) est l'arythmie cardiaque la plus fréquente et l'une des principales causes d'AVC. Elle peut être détectée à partir d'un simple enregistrement ECG court, notamment via des dispositifs portables grand public (ex : AliveCor/KardiaMobile). Ce projet s'appuie sur le dataset du **PhysioNet/CinC Challenge 2017**, qui propose des enregistrements ECG mono-dérivation courts (9 à 61 secondes), classés en 4 catégories :

- **Normal** — rythme sinusal normal
- **AF** — fibrillation auriculaire
- **Other** — autre rythme anormal
- **Noisy** — signal trop bruité pour être classifié

## Objectifs du projet

- [ ] Comprendre le dataset et la problématique clinique de la FA
- [ ] Charger et explorer les enregistrements ECG (visualisation, statistiques des classes)
- [ ] Implémenter un pipeline de prétraitement du signal (filtrage, nettoyage du bruit)
- [ ] Extraire des features pertinentes (ex : variabilité du rythme cardiaque, features spectrales)
- [ ] Entraîner et évaluer un modèle de classification (ex : Random Forest, CNN)
- [ ] Comparer les résultats obtenus à ceux de la littérature de référence
- [ ] Interpréter les résultats (matrice de confusion, F1-score par classe, importance des features)

## Structure du dépôt

```
.
├── data/                   # Données brutes et prétraitées (non versionnées, voir .gitignore)
├── notebooks/              # Notebooks d'exploration et de prototypage
├── src/                    # Code source du pipeline
│   ├── preprocessing.py    # Filtrage et nettoyage des signaux ECG
│   ├── features.py         # Extraction de features
│   ├── model.py             # Entraînement et évaluation du modèle
│   └── utils.py             # Fonctions utilitaires
├── results/                 # Résultats, figures, métriques
├── report/                  # Rapport final du projet
├── requirements.txt         # Dépendances Python
└── README.md
```

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd <nom-du-repo>

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Données

Le dataset est disponible publiquement sur PhysioNet :
[https://physionet.org/content/challenge-2017/1.0.0/](https://physionet.org/content/challenge-2017/1.0.0/)

```bash
# Exemple de téléchargement (à adapter selon la méthode choisie)
wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/
```

> Les données ne sont pas versionnées dans ce dépôt (voir `.gitignore`). Merci de les télécharger séparément et de les placer dans le dossier `data/`.

## Utilisation

```bash
# Exploration des données
jupyter notebook exploring/exploration.ipynb

# Lancer le pipeline complet
jupyter notebook exploring/preprocessing.ipynb
```

## Méthodologie

1. **Prétraitement** : filtrage passe-bande, suppression du bruit, normalisation
2. **Extraction de features** : détection des pics R, calcul de la variabilité du rythme cardiaque (HRV), features spectrales
3. **Modélisation** : entraînement d'un modèle de classification (à préciser selon l'avancée du projet)
4. **Évaluation** : validation croisée, métriques adaptées au déséquilibre des classes (F1-score macro, comme dans le challenge original)

## Résultats

*en moyenne*

| Modèle | Accuracy | F1-score (macro) |
|---|---|---|
| Baseline | 0.87 | 78.08 |

## Références

- PhysioNet/CinC Challenge 2017 : *AF Classification from a Short Single Lead ECG Recording*. [https://physionet.org/content/challenge-2017/1.0.0/](https://physionet.org/content/challenge-2017/1.0.0/)
- Clifford, G. D., et al. (2017). *AF Classification from a Short Single Lead ECG Recording: The PhysioNet Computing in Cardiology Challenge 2017*. Computing in Cardiology.

## Équipe

- [Nom Prénom]
- [Nom Prénom]

## Licence

Projet académique — usage éducatif uniquement.
