# YOLOv11 RT-DETR ROC calibration ablation

## Objectif

Cette ablation vise a ameliorer la probabilite de detection au point de fonctionnement impose par un taux de fausse alarme cible.

Elle s'applique a un modele YOLOv11 deja entraine avec une tete one-to-one RT-DETR. Le modele complet est charge depuis un `best.pt`, puis seuls les scores de classification de la tete RT-DETR sont raffines. Le backbone, le neck, le decodeur et les tetes de regression restent figes.

## Theorie

Dans une tete de detection de type RT-DETR, le decodeur produit un ensemble fixe de requetes de detection. Chaque requete `i` predit :

```text
(b_i, s_i)
```

avec :

- `b_i` : boite predite par la requete.
- `s_i in [0, 1]` : score de confiance associe a la classe predite.

Dans l'implementation, `s_i` est obtenu a partir des logits de classification de la tete RT-DETR. Pour une requete positive, on prend le score de la classe de la cible matchee. Pour une requete negative, on prend le plus grand score de classe, car c'est ce score maximal qui peut declencher une fausse alarme.

Apres l'appariement hongrois, les requetes sont partitionnees en deux ensembles :

```text
P = {i | la query i est associee a une cible reelle}
N = {i | la query i n'est associee a aucune cible}
```

L'ensemble `P` regroupe donc les hypotheses positives, c'est-a-dire les requetes qui doivent detecter un objet reel. L'ensemble `N` regroupe les hypotheses de fond. Ce sont ces requetes negatives qui controlent directement le risque de fausse alarme, puisqu'une requete negative avec un score eleve peut etre conservee apres seuillage.

L'objectif de cette ablation n'est pas seulement d'ameliorer la loss moyenne de classification. L'objectif est d'ameliorer la probabilite de detection autour d'un point de fonctionnement impose par un taux de fausse alarme cible :

```text
Pfa*
```

Pour cela, on estime un seuil de decision `tau*` a partir de la distribution des scores des requetes negatives :

```text
tau* = stopgrad[Q_{1 - Pfa*}({s_i, i in N})]
```

ou :

- `Q_{1 - Pfa*}` designe le quantile empirique d'ordre `1 - Pfa*`.
- `{s_i, i in N}` est l'ensemble des scores des requetes negatives.
- `stopgrad` signifie que le seuil est considere comme une constante pendant la retropropagation.

Par construction, une proportion approximativement egale a `Pfa*` des requetes negatives possede un score superieur a `tau*`. Par exemple, si `Pfa* = 0.01`, alors `tau*` correspond au quantile 0.99 des scores negatifs : environ 1 % des requetes negatives sont au-dessus du seuil.

Ce seuil represente donc le niveau de score qu'une detection doit depasser pour etre visible au point de fonctionnement vise. On ajoute alors une fonction de cout additionnelle qui pousse les requetes positives au-dessus de ce seuil :

```text
L_ROC = (1 / |P|) sum_{i in P} softplus(alpha * (tau* + m - s_i))
```

avec :

- `|P|` : nombre de requetes positives.
- `m >= 0` : marge de securite au-dessus du seuil de fausse alarme.
- `alpha > 0` : raideur de la penalisation.
- `softplus(x) = log(1 + exp(x))`.

Cette perte penalise fortement une requete positive lorsque son score `s_i` est inferieur au seuil cible `tau* + m`. Elle devient faible lorsque :

```text
s_i >= tau* + m
```

L'interpretation est la suivante : les scores positifs ne sont pas seulement encourages a etre grands dans l'absolu, ils sont encourages a etre plus grands que le niveau de score atteint par les fausses alarmes les plus competitives.

Cette formulation agit donc comme une calibration locale de la courbe ROC autour du taux de fausse alarme cible. Elle cherche a augmenter la separation entre :

- les requetes positives matchees a des objets reels ;
- la queue haute de la distribution des scores negatifs.

## Loss de raffinement

La loss utilisee pendant cette phase est :

```text
L_fine = L_cls + mu * L_ROC
```

avec :

- `L_cls` : loss de classification RT-DETR standard, actuellement Varifocal Loss.
- `L_ROC` : loss de calibration au seuil de fausse alarme cible.
- `mu` : poids de la loss ROC.

Les pertes de localisation ne sont pas utilisees pendant cette phase :

```text
L_bbox = 0
L_giou = 0
```

Le but est de recalibrer les scores sans modifier les boites apprises pendant l'entrainement principal.

## Parametres entraines

Le script fige tous les parametres, puis rend trainables uniquement :

- `detect_one2one.enc_score_head`
- `detect_one2one.dec_score_head`

Tout le reste reste fige :

- backbone YOLOv11
- neck
- decodeur RT-DETR
- references de boites
- tetes bbox
- DFL

## Script

Le script d'experience est :

```text
core/scripts/run_yolov11_rtdetr_roc_calibration.py
```

Exemple :

```bash
python core/scripts/run_yolov11_rtdetr_roc_calibration.py \
  --weights /data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_ablation/yolov11n_best_ft_rtdetr_head_one2one_deformable/best.pt \
  --pfa-target 0.01 \
  --roc-margin 0.02 \
  --roc-alpha 20 \
  --roc-weight 1.0 \
  --epochs 30
```

Sortie par defaut :

```text
/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_ablation/yolov11n_rtdetr_head_roc_calibration
```

## Interpretation des hyperparametres

- `--pfa-target` : proportion cible de queries negatives autorisees au-dessus du seuil `tau`.
- `--roc-margin` : marge ajoutee au seuil pour pousser les positifs plus loin que le niveau de fausse alarme.
- `--roc-alpha` : raideur de la penalisation `softplus`.
- `--roc-weight` : coefficient `mu` devant `L_ROC`.
- `--no-aux-cls-loss` : desactive l'application de la calibration sur les sorties auxiliaires du decodeur.

## Evaluation

L'evaluation conserve la logique existante du projet :

- le modele produit des predictions avec scores ;
- `dataset_analysis_with_metrics` choisit un seuil de confiance correspondant a `fa_target` ;
- les metriques de detection et les recalls SNR sont mesurees a ce point de fonctionnement.

L'ablation cherche donc a rendre la distribution des scores plus favorable autour de ce seuil.

## Pourquoi ne pas l'appliquer a TAL pour l'instant

La formulation est directe pour RT-DETR car le matching hongrois fournit une partition nette des queries :

```text
positives = queries matchees
negatives = queries non matchees
```

Avec une tete YOLO one-to-one TAL, l'assignation repose sur des ancres/grilles multi-echelles et sur des scores de qualite TAL. La notion de negative query comparable a celle de RT-DETR est moins propre, et la calibration peut interferer avec l'assignation task-aligned.

Pour cette raison, l'ablation actuelle est limitee a la tete RT-DETR one-to-one.
