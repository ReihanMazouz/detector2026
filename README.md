# detector2026

Projet de detection RF avec entrainement YOLO, API FastAPI et interface frontend React/Vite.

## Structure

- `core/` contient les modeles, dataloaders, preprocessings et scripts d'entrainement.
- `backend/` expose l'API FastAPI pour le pilotage des datasets, de l'entrainement et de l'evaluation.
- `frontend/` contient l'interface React/Vite.
- `runs/` sert aux sorties locales d'entrainement et d'evaluation.

## Installation

Le projet est prevu pour fonctionner avec un environnement Conda local dans `./.conda-env`.

```bash
conda env create --prefix ./.conda-env -f environment.yml
```

Pour le frontend :

```bash
cd frontend
npm install
cd ..
```

## Lancement local

Lancer backend + frontend :

```bash
zsh start_dev.sh
```

Lancer uniquement l'API backend :

```bash
zsh start_backend.sh
```

Par defaut :

- backend : `http://127.0.0.1:8001`
- frontend : `http://127.0.0.1:5174`

## Entrainement

Scripts actuellement presents dans `core/scripts/` :

- `train_multi_res.py` : entrainement multi-resolution avec `MR_YOLO` sur dataset `fused`
- `train_unires.py` : entrainement uni-resolution avec `YOLOv8` sur dataset `specificres`
- `train_complex_unires_yolov11.py` : entrainement uni-resolution avec `YOLOv11` sur spectres complexes

Les preprocessings disponibles sont centralises dans `core/utils/preprocess/`.

Exemple :

```bash
python core/scripts/train_complex_unires_yolov11.py
```

Les chemins dataset, sorties et hyperparametres des scripts d'entrainement sont actuellement codes en dur dans les fichiers.

## API

Le backend expose notamment :

- `/health`
- `/dataset/stats`
- `/dataset/examples`
- `/training/models`
- `/training/start`
- `/training/runs`
- `/evaluation/runs`

Le point d'entree principal est `backend/app.py`.

## Git

Le depot `detector2026` est volontairement separe du depot parent `icml`.

Workflow minimal :

```bash
git status
git add .
git commit -m "Your message"
git push
```
