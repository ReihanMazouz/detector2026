# Classical Detectors

Baselines de détection RF classiques sur **signal 1D réel complet**, intégrées dans `detector2026`.

Le périmètre est volontairement simple :
- détection binaire `H0/H1` sur un signal entier,
- calibration de la `Pfa`,
- évaluation de la probabilité de détection `Pd` en fonction du `SNR`,
- cadre limité au dataset **single emitter**.

## Détecteurs disponibles

- `quadratic` : détecteur quadratique classique à seuil analytique.
- `fft_1d` : détecteur par maximum spectral sur la FFT 1D, avec seuil analytique.
- `tf_glrt` : GLRT cellule par cellule sur le spectre temps-fréquence, avec seuil calibré sur bruit simulé.
- `qmf` : détecteur QMF par analyse multi-résolution, avec seuil calibré sur bruit simulé.

## Structure

```text
baselines/classical_detectors/
├── detectors/      # Détecteurs
├── evaluation/     # Bruit, calibration Pfa, métriques, benchmark
├── io/             # Chargement du dataset single emitter
└── scripts/        # Scripts de validation et benchmark
```

## Hypothèses

- pas de multi-émetteurs,
- pas de post-traitement,
- pas de fenêtre glissante,
- pas de comparaison YOLO dans ce module,
- bruit additif blanc gaussien réel, avec variance fixe par sample.
- `SNR = (sum(s^2) / duration_samples) / noise_variance`, cohérent avec le SNR en puissance du simulateur.

## Dataset attendu

Par défaut, les scripts pointent vers :

`/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/tmp/output/rf_single_emitter_validation_smoketest_f32`

Le dataset doit contenir :
- `raw_data/*.pt`
- `labels_detect/*.json`
- `representation_config.json`

Pour le sweep par forme d'onde non bruite, `run_waveform_snr_sweep.py` attend aussi le layout :

```text
<dataset_root>/
├── dataset_config.json
├── manifest.json
└── <waveform_label>/scenario_00000/{signal.pt,pulse.json}
```

## Commandes utiles

Depuis la racine de `detector2026` :

Evaluation simple sur le dataset `manifest.json` non bruite :

```bash
python baselines/classical_detectors/scripts/evaluate_dataset.py
```

Les chemins du dataset et des poids `best.pt` sont des constantes en haut du script.
La normalisation deep appliquee est `log_snr_estimated`, comme dans `ICML2026DataSimulator`.
Pour les signaux reels, les spectres deep et le `tf_glrt` utilisent la `rfft`.

Par defaut, le script evalue aussi `quadratic`, `fft_1d`, `tf_glrt` et `qmf`, calibre chaque methode a la meme `Pfa`, et ecrit `runs/baselines/dataset_evaluation.json`.

```bash
/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/.conda-env/bin/python \
baselines/classical_detectors/scripts/run_noise_validation.py \
--n-trials 5000 --pfa 0.01
```

```bash
/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/.conda-env/bin/python \
baselines/classical_detectors/scripts/run_single_emitter_benchmark.py \
--noise-trials 5000 --pfa 0.01
```

```bash
/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/.conda-env/bin/python \
baselines/classical_detectors/scripts/run_waveform_snr_sweep.py \
--dataset-root runs/classical_detectors/datasets/single_emitter_noiseless \
--pfa 0.001 \
--noise-trials 5000 \
--noise-variance 1.0 \
--snr-values-db=-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10
```

## Sorties

Les résultats sont écrits par défaut dans :

- `runs/classical_detectors/noise_validation.json`
- `runs/classical_detectors/single_emitter_benchmark.json`
- `runs/classical_detectors/waveform_snr_sweep.json`
- `runs/classical_detectors/waveform_snr_sweep_by_snr.png`

## Remarque sur la Pfa FFT

Le seuil FFT actuellement implémenté est analytique. Sur un faible nombre de tirages bruit, la `Pfa` empirique peut fluctuer sensiblement autour de la cible. Pour une validation scientifique correcte, utiliser plusieurs milliers de tirages.
