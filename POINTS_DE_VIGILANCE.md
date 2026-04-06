# Points De Vigilance

Cette note centralise des points connus dans le projet qui peuvent biaiser les analyses
ou meriter une correction ulterieure.

## Evaluation detection

### 1. Une prediction ne peut pas couvrir plusieurs GT

- Dans l'evaluation actuelle, le matching detection/ground truth est fait en mode
  `1 prediction <-> 1 GT`.
- Donc une seule boite predite ne peut pas compter comme TP pour plusieurs GT,
  meme si elle recouvre effectivement plusieurs objets avec une IoU superieure au seuil.
- Point de code principal:
  - `core/utils/metrics.py::match_boxes_iou`
  - `core/utils/analysing_results.py::analyse_results`
- Impact:
  - cela peut penaliser certaines fusions ou certains cas de scenes denses
  - un oracle mal formule peut perdre artificiellement du recall si la meme prediction
    est reutilisee pour plusieurs GT

### 2. La "fausse alarme" est obtenue apres post-process, par filtrage sur le score

- Dans `dataset_analysis_with_metrics`, les stats brutes sont d'abord calculees avec
  un `postprocess` fait a seuil bas (`conf_thresh=0.05`), puis le seuil correspondant
  a la cible de "fausse alarme" est choisi apres coup en filtrant les TP/FP par score.
- Le seuil n'est donc pas obtenu en relancant directement le `postprocess` du modele
  pour chaque valeur de confiance candidate.
- Point de code principal:
  - `core/utils/analysing_results.py::dataset_analysis_with_metrics`
- Impact:
  - ce n'est pas strictement equivalent a une vraie reevaluation complete du modele
    pour chaque seuil de confiance
  - cela peut creer des ecarts si un autre script applique le seuil directement dans
    le `postprocess` au lieu de reproduire la logique standard du projet

## Remarque

- Dans l'etat actuel du projet, quand on compare des runs individuels, des fusions
  (`oracle`, `nms_fusion`) ou des scripts d'analyse externes, il faut verifier que
  la convention d'evaluation utilisee est exactement la meme avant d'interpreter les
  differences de recall, mAP ou matrices de confusion.
