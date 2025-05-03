
# Skin Burn Segmentation

Un projet de détection et segmentation de brûlures cutanées à partir du **Skin Burn Dataset**, implémenté en PyTorch et `segmentation_models_pytorch`, et optimisé pour tourner sur GPU limité (Tesla T4).

---

## Sommaire

- [Contexte](#contexte)  
- [Fonctionnalités principales](#fonctionnalités-principales)  
- [Jeu de données](#jeu-de-données)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Structure du projet](#structure-du-projet)  
- [Pipeline de traitement](#pipeline-de-traitement)  
- [Résultats et métriques](#résultats-et-métriques)  
- [Contribuer](#contribuer)  
- [Licence](#licence)  

---

## Contexte

Ce projet vise à automatiser la détection et la segmentation des zones de brûlure sur des images cliniques.  
Il s’appuie sur un réseau de type **U‑Net** avec différents encodeurs (SE‑ResNeXt50, ResNet50, ResNet18) et intègre :  
- Une validation croisée K‑fold  
- De la précision mixte (AMP)  
- Des callbacks avancés (EarlyStopping, CosineAnnealingWarmRestarts, ReduceLROnPlateau)  
- Un pipeline d’augmentation via Albumentations  

---

## Fonctionnalités principales

1. **Chargement et parsing**  
   - Images JPEG et annotations YOLO (.txt)  
2. **Data augmentation**  
   - Flips, rotations, contrast, etc. (Albumentations)  
3. **Validation croisée K‑fold**  
   - Par défaut 3 folds, paramétrable  
4. **Modèles U‑Net**  
   - Encoders : `se_resnext50_32x4d`, `resnet50`, `resnet18`  
5. **Entraînement en précision mixte (AMP)**  
   - Réduction de l’usage de la VRAM  
6. **Callbacks**  
   - EarlyStopping, scheduler CosineAnnealingWarmRestarts, ReduceLROnPlateau  
7. **Sauvegarde automatique**  
   - Meilleurs poids par fold  
8. **Évaluation**  
   - IoU, précision globale et métriques par classe  

---

## Jeu de données

- **Origine** : hébergé sur Kaggle par Shubham Baid sous licence CC0  
- **Contenu** : ~1 300 images JPEG + fichiers d’annotations au format YOLO  
- **Classes** :  
  - `0` – Brûlure de 1er degré  
  - `1` – Brûlure de 2ᵉ degré  
  - `2` – Brûlure de 3ᵉ degré  

Chaque `.txt` porte le même nom que l’image correspondante et contient, pour chaque objet annoté :
```text
<classe> <x_center> <y_center> <largeur> <hauteur>
````

---

## Installation

> **Prérequis**
>
> * Python ≥ 3.8
> * CUDA et GPU (recommandé GPU ≥ 8 Go VRAM)

```bash
# Clone du dépôt
git clone https://github.com/VOTRE_UTILISATEUR/skin-burn-segmentation.git
cd skin-burn-segmentation

# Création et activation d’un environnement virtuel
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Installation des dépendances
pip install -r requirements.txt
```

---

## Usage

1. **Préparez les données**

   * Placez les dossiers `images/` et `labels/` dans `data/`.
2. **Configuration**

   * Ajustez les paramètres (chemins, hyperparamètres, folds, batch size) dans `config.yaml`.
3. **Lancement de l’entraînement**

   ```bash
   python train.py --config config.yaml
   ```
4. **Évaluation**

   ```bash
   python evaluate.py --weights path/to/best_model.pth --config config.yaml
   ```

---

## Structure du projet

```
.
├── data/                    # Données brutes (images + labels)
├── src/
│   ├── datasets.py          # Chargement et parsing YOLO
│   ├── augmentations.py     # Définitions Albumentations
│   ├── models.py            # U‑Net et choix d’encodeurs
│   ├── train.py             # Boucle d’entraînement
│   ├── evaluate.py          # Scripts d’évaluation
│   └── utils.py             # Callbacks, métriques, visualisations
├── config.yaml              # Configuration générale
├── requirements.txt         # Dépendances Python
└── README.md                # Ce document
```

---

## Pipeline de traitement

1. **Lecture et prétraitement**

   * Redimensionnement à 384×384 px
   * Normalisation
2. **Data augmentation**
3. **K‑fold splitting**
4. **Entraînement**

   * Précision mixte (AMP)
   * Scheduler CosineAnnealingWarmRestarts
5. **Validation**

   * Calcul de la loss et des métriques (IoU, precision) par fold
6. **Agrégation des résultats**

   * Moyennes et écarts‑types sur les folds

---

## Résultats et métriques

Après 3 folds de validation croisée :

* **Perte moyenne** : 0.1971 ± 0.0153
* **IoU moyen** : 0.6939 ± 0.0134
* **Précision moyenne** : 91.52 % ± 0.57 %

**Performances par encodeur** :

* `se_resnext50_32x4d` : IoU finale ≃ 0.7252, Accuracy ≃ 92.85 %
* `resnet50` (folds 2 & 3) : IoU ≃ 0.67–0.69, Accuracy ≃ 90–91 %
* `resnet18` (prototype léger) : entraînements plus instables

---

## Contribuer

Les contributions sont les bienvenues !

1. Forkez ce dépôt
2. Créez une branche (`git checkout -b feature/ma-fonctionnalite`)
3. Commitez vos modifications (`git commit -m 'Ajout de ...'`)
4. Poussez votre branche (`git push origin feature/ma-fonctionnalite`)
5. Ouvrez une Pull Request

---

## Licence

Ce projet est mis à disposition sous licence **MIT**.
Voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
```
