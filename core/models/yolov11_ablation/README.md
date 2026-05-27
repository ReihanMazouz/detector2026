# Ablations YOLOv11 RT-DETR

Ce dossier regroupe les ablations YOLOv11 avec des modules transformers. 

## 1. YOLOv11RTDETRHead

Le modèle contient deux têtes :

- `detect` : tête YOLO one-to-many originale, utilisée pour charger les poids du checkpoint YOLOv11.
- `detect_one2one` : tête RT-DETR one-to-one, utilisée pendant l'entraînement de cette ablation.

Lors du fine-tuning one-to-one, `train_one2one_head_only()` gèle le backbone, le neck et la tête YOLO, puis entraîne uniquement `detect_one2one`. La branche bbox YOLO est copiée vers la tête RT-DETR :

```text
detect.cv_dist -> detect_one2one.cv_dist
detect.dfl     -> detect_one2one.dfl
```

### 1.1. Sélection Des Queries

Soit \(F \in \mathbb{R}^{B \times N \times C}\) l'ensemble des tokens multi-échelles aplatis après projection, avec :

- \(B\) : taille du batch ;
- \(N\) : nombre total de tokens issus de P3/P4/P5 ;
- \(C\) : dimension cachée `hidden_dim`.

La tête de score encodeur prédit des logits de classe :

\[
S = h_{cls}(F), \quad S \in \mathbb{R}^{B \times N \times K}
\]

où \(K\) est le nombre de classes. Dans le chemin RT-DETR avec varifocal loss, il n'y a pas de classe explicite `no-object`.

Les `num_queries` meilleurs tokens sont sélectionnés par :

\[
i_q = \operatorname{TopK}_q \left( \max_{k \in \{1,\dots,K\}} S_{:, :, k} \right)
\]

Les features sélectionnées deviennent les queries du decoder. Les boîtes YOLO/DFL correspondantes deviennent les boîtes de référence initiales.

### 1.2. Raffinement Itératif Des Boîtes

Le decoder RT-DETR contient `num_decoder_layers` couches. Elles s'enchaînent : la couche \(l+1\) utilise les boîtes raffinées par la couche \(l\).

Pour la couche decoder \(l\), on note :

- \(z_l\) : embedding de query après la couche \(l\) ;
- \(b_{l-1} \in [0,1]^4\) : boîte de référence issue de la couche précédente ;
- \(\Delta b_l = h_{box,l}(z_l)\) : correction de boîte prédite par la couche \(l\).

La boîte raffinée est :

\[
b_l = \sigma\left(\Delta b_l + \sigma^{-1}(b_{l-1})\right)
\]

avec :

- \(\sigma\) : fonction sigmoid ;
- \(\sigma^{-1}\) : inverse sigmoid ;
- \(b_l\) : boîte normalisée au format \((x_c, y_c, w, h)\).

Pendant l'entraînement, la couche suivante utilise :

\[
b_l^{ref} = \operatorname{detach}(b_l)
\]

La couche suivante voit donc bien la boîte raffinée, mais son gradient ne remonte pas à travers les coordonnées de la couche précédente. C'est le comportement utilisé dans le decoder RT-DETR d'Ultralytics/DEYO.

### 1.3. Matching Hungarian Et Loss

Chaque sortie decoder prédit :

\[
(\hat{p}, \hat{b}) = (\text{logits de classe}, \text{boîte})
\]

Le matching est one-to-one : chaque objet ground truth est assigné à au plus une query par algorithme Hungarian. Le coût combine classification focal, distance L1 et GIoU :

\[
\mathcal{C} = \lambda_{cls}\mathcal{C}_{cls}
            + \lambda_{box}\lVert \hat{b} - b \rVert_1
            + \lambda_{giou}\left(1 - \operatorname{GIoU}(\hat{b}, b)\right)
\]

où :

- \(\hat{b}\) : boîte prédite ;
- \(b\) : boîte ground truth ;
- \(\lambda_{cls}\), \(\lambda_{box}\), \(\lambda_{giou}\) : poids des trois termes.

La classification utilise une cible de type varifocal. Pour une prédiction matchée, le score cible de la classe est l'IoU avec la boîte ground truth associée. Les autres classes ont une cible nulle.

### 1.4. Supervision Auxiliaire

La loss d'entraînement n'est pas appliquée uniquement à la dernière couche decoder. Elle est aussi appliquée aux sorties intermédiaires et à la sortie encodeur top-k :

\[
\mathcal{L}_{total}
= \mathcal{L}_{dec,L}
+ \mathcal{L}_{enc}
+ \sum_{l=1}^{L-1} \mathcal{L}_{dec,l}
\]

où :

- \(L\) : nombre de couches decoder ;
- \(\mathcal{L}_{dec,L}\) : loss de la dernière couche decoder ;
- \(\mathcal{L}_{dec,l}\) : loss auxiliaire de la couche decoder \(l\) ;
- \(\mathcal{L}_{enc}\) : loss sur les boîtes et logits sélectionnés par l'encodeur.

Chaque terme \(\mathcal{L}\) contient :

\[
\mathcal{L} =
\lambda_{cls}\mathcal{L}_{cls}
+ \lambda_{box}\mathcal{L}_{L1}
+ \lambda_{giou}\mathcal{L}_{GIoU}
\]

Cette supervision auxiliaire stabilise l'entraînement du transformer, mais elle augmente la valeur numérique de la loss car plusieurs termes sont additionnés.

### 1.5. Post-Process D'Inférence

Pour la tête RT-DETR one-to-one :

1. les logits de classe sont convertis avec sigmoid ;
2. la meilleure classe et son score sont sélectionnés pour chaque query ;
3. les prédictions sous le seuil de confiance sont supprimées ;
4. les boîtes sont converties de \((x_c, y_c, w, h)\) normalisé vers \((x_1, y_1, x_2, y_2)\) en pixels ;
5. aucun NMS n'est appliqué.

RT-DETR est one-to-one par construction : la suppression des doublons doit être apprise par le modèle plutôt qu'ajoutée par NMS.

## 2. YOLOv11RTDETR

`YOLOv11RTDETR` est la version la plus proche d'un RT-DETR complet dans cette ablation. Elle conserve le backbone YOLOv11 jusqu'aux cartes \(P_3, P_4, P_5\), remplace le neck FPN/PAN YOLO par un neck hybride RT-DETR, puis applique la tête `RTDETRHead`.

Le chemin est :

```text
image -> backbone YOLOv11 -> P3, P4, P5 -> RTDETRHybridEncoderNeck -> RTDETRHead
```

Contrairement à `YOLOv11RTDETRHead`, cette variante n'utilise pas le neck YOLO. Les modules FPN/PAN et la tête YOLO standard sont gelés et inutilisés pendant le forward.

### 2.1. Neck Hybride RT-DETR

Le `RTDETRHybridEncoderNeck` suit une logique hybride : il ne fait pas une self-attention globale sur tous les tokens P3/P4/P5. Il projette d'abord les trois niveaux vers une dimension commune, encode seulement le niveau le plus profond \(P_5\) avec un transformer, puis refusionne les niveaux par un chemin FPN/PAN convolutionnel.

Chaque carte multi-échelle est d'abord projetée vers une dimension commune \(C\), égale à `hidden_dim` :

\[
X_l = \phi_l(P_l), \quad X_l \in \mathbb{R}^{B \times C \times H_l \times W_l}, \quad l \in \{3,4,5\}
\]

où :

- \(P_l\) est la carte backbone au niveau \(l\) ;
- \(\phi_l\) est une projection \(1 \times 1\) ;
- \(B\) est la taille du batch ;
- \(C\) est la dimension cachée RT-DETR.

Seul \(X_5\), le niveau le plus bas en résolution et le plus riche sémantiquement, est envoyé dans un `TransformerEncoder`. Il est d'abord converti en séquence :

\[
T_5 = \operatorname{Flatten}(X_5) + e(x,y),
\quad
T_5 \in \mathbb{R}^{B \times N_5 \times C}
\]

avec :

- \(N_5 = H_5W_5\) le nombre de tokens du niveau \(P_5\) ;
- \(e(x,y)\) une projection linéaire des coordonnées spatiales normalisées \((x,y)\).

Le transformer encode uniquement ces tokens :

\[
Z_5 = \operatorname{TransformerEncoder}(T_5)
\]

puis la séquence est remise sous forme de carte :

\[
\tilde{X}_5 = \operatorname{Unflatten}(Z_5),
\quad
\tilde{X}_5 \in \mathbb{R}^{B \times C \times H_5 \times W_5}
\]

Cette attention est donc coûteuse en \(\mathcal{O}(N_5^2 C)\), et non en \(\mathcal{O}((N_3+N_4+N_5)^2 C)\). C'est un point important : le neck hybride RT-DETR est beaucoup moins coûteux qu'une attention dense sur tous les tokens P3/P4/P5.

Les informations sont ensuite propagées vers les hautes résolutions par un chemin FPN top-down :

\[
U_4 = \operatorname{C3k2}\left([ \operatorname{Up}(\operatorname{Conv}_{1\times1}(\tilde{X}_5)); X_4]\right)
\]

\[
Y_3 = \operatorname{C3k2}\left([ \operatorname{Up}(\operatorname{Conv}_{1\times1}(U_4)); X_3]\right)
\]

où \([\,;\,]\) désigne une concaténation canal et \(\operatorname{Up}\) une interpolation nearest-neighbor de facteur 2.

Un chemin PAN bottom-up refusionne ensuite les informations vers les niveaux plus profonds :

\[
Y_4 = \operatorname{C3k2}\left([ \operatorname{Down}(Y_3); U_4]\right)
\]

\[
Y_5 = \operatorname{C3k2}\left([ \operatorname{Down}(Y_4); \operatorname{Conv}_{1\times1}(\tilde{X}_5)]\right)
\]

où \(\operatorname{Down}\) est une convolution \(3 \times 3\) de stride 2.

Le neck renvoie finalement :

\[
(Y_3, Y_4, Y_5) = \operatorname{HybridEncoder}(X_3, X_4, X_5)
\]

Ces cartes ont toutes \(C\) canaux et servent directement d'entrée à `RTDETRHead`. Le rôle du neck est donc de produire des features multi-échelles alignées dans l'espace latent RT-DETR, en combinant attention globale sur \(P_5\) et fusion locale multi-échelle par convolutions.

### 2.2. Tête RT-DETR

La tête applique ensuite le même schéma que `YOLOv11RTDETRHead` :

\[
F = \operatorname{FlattenConcat}(Y_3, Y_4, Y_5), \quad F \in \mathbb{R}^{B \times N \times C}
\]

où \(N = H_3W_3 + H_4W_4 + H_5W_5\). Les `num_queries` meilleurs tokens sont sélectionnés par score encodeur, puis raffinés par le decoder déformable :

\[
b_l = \sigma\left(\Delta b_l + \sigma^{-1}(b_{l-1})\right)
\]

avec \(b_l\) la boîte normalisée prédite à la couche decoder \(l\), et \(\Delta b_l\) la correction prédite par la tête bbox de cette couche. La loss reste une loss one-to-one avec matching Hungarian, classification varifocal, L1 et GIoU.

## 3. YOLOv11P3Direct Et YOLOv11P3RTDETR

Cette ablation isole l'effet d'une tête RT-DETR lorsque le modèle ne dispose que d'un seul niveau de features. Elle compare trois entraînements construits sur le même backbone YOLOv11n simplifié, arrêté au niveau \(P_3\).

Le script associé est :

```text
core/scripts/run_yolov11_p3_rtdetr_ablation.py
```

Les sorties sont :

```text
exp1_yolov11n_p3_direct_tal_topk10
exp2_1_yolov11n_p3_rtdetr_frozen_backbone
exp2_2_yolov11n_p3_rtdetr_full_train
```

### 3.1. Expérience 1 : YOLOv11P3Direct

`YOLOv11P3Direct` conserve uniquement le début du backbone YOLOv11 jusqu'à \(P_3\), puis applique directement une tête YOLO `Detect` mono-échelle. Le neck FPN/PAN, \(P_4\), \(P_5\), `SPPF` et `C2PSA` sont supprimés.

Le chemin est :

```text
image -> conv1 -> conv2 -> C3k2 -> conv3 -> C3k2 -> P3 -> Detect
```

Pour une entrée \(256 \times 256\), le niveau \(P_3\) a typiquement :

\[
P_3 \in \mathbb{R}^{B \times 64 \times 32 \times 32}
\]

avec la configuration `width_mult=0.25`. La tête YOLO produit :

\[
\hat{D}_3 \in \mathbb{R}^{B \times 4r \times 32 \times 32},
\quad
\hat{S}_3 \in \mathbb{R}^{B \times K \times 32 \times 32}
\]

où :

- \(r=\texttt{reg\_max}\) est le nombre de bins DFL par côté de boîte ;
- \(K\) est le nombre de classes ;
- \(\hat{D}_3\) contient les distributions de distances DFL ;
- \(\hat{S}_3\) contient les logits de classification.

La loss est la loss YOLO classique avec `TaskAlignedAssigner` :

\[
\mathcal{L}_{YOLO}
=
\lambda_{box}\mathcal{L}_{box}
+ \lambda_{cls}\mathcal{L}_{BCE}
+ \lambda_{dfl}\mathcal{L}_{DFL}
\]

L'assignation utilise explicitement :

```text
tal_topk = 10
```

Cette expérience sert de référence : elle apprend un détecteur P3 simple, puis son `best.pt` initialise les deux variantes RT-DETR suivantes.

### 3.2. Expérience 2.1 : P3 RT-DETR Avec Backbone Figé

`YOLOv11P3RTDETR` reprend le même backbone P3 que l'expérience 1, charge :

```text
exp1_yolov11n_p3_direct_tal_topk10/best.pt
```

puis remplace la tête YOLO entraînable par une tête `RTDETRHead` mono-échelle. Dans l'expérience 2.1, le backbone est figé et seule la tête RT-DETR est entraînée.

Le chemin forward est :

```text
image -> backbone P3 figé -> P3 -> RTDETRHead
```

La feature \(P_3\) est projetée vers la dimension cachée \(d\) :

\[
X_3 = \phi_3(P_3),
\quad
X_3 \in \mathbb{R}^{B \times d \times H_3 \times W_3}
\]

puis aplatie :

\[
M = \operatorname{Flatten}(X_3),
\quad
M \in \mathbb{R}^{B \times N_3 \times d},
\quad
N_3 = H_3W_3
\]

Les `num_queries` meilleurs tokens sont sélectionnés par le score encodeur :

\[
i_q =
\operatorname{TopK}_q
\left(
\max_{k \in \{1,\dots,K\}} h_{cls}(M)_{:, :, k}
\right)
\]

La tête utilise une attention déformable mono-niveau au sens des cartes mémoire : le decoder ne voit que \(P_3\), donc `num_levels=1`. Cela ne signifie pas une seule couche de decoder. Dans le script, `num_decoder_layers=6` par défaut ; ces couches s'enchaînent avec raffinement itératif des boîtes.

Pour une couche de decoder donnée, la cross-attention devient :

\[
\operatorname{MSDeformAttn}(q_i)
=
\sum_{k=1}^{K_p}
a_{i,k}\,
V_3(p_i + \Delta p_{i,k})
\]

où :

- \(q_i\) est la query \(i\) ;
- \(K_p=\texttt{num\_decoder\_points}=16\) est le nombre de points échantillonnés ;
- \(p_i\) est la boîte ou position de référence de la query ;
- \(\Delta p_{i,k}\) est l'offset appris du point \(k\) ;
- \(a_{i,k}\) est le poids d'attention appris ;
- \(V_3\) est la carte de valeurs issue de \(P_3\).

À chaque couche \(l\), la boîte est raffinée par :

\[
b_l = \sigma\left(\Delta b_l + \sigma^{-1}(b_{l-1})\right)
\]

En entraînement, la couche suivante utilise bien \(b_l\) comme nouvelle référence, avec un `detach` sur ces coordonnées.

Cette expérience mesure la capacité d'une tête RT-DETR à exploiter des features P3 déjà apprises, sans modifier l'extracteur.

### 3.3. Expérience 2.2 : P3 RT-DETR En Full Training

L'expérience 2.2 utilise la même initialisation depuis `best.pt`, la même tête `RTDETRHead` et la même attention déformable à 16 points, mais ne fige pas le backbone :

```text
image -> backbone P3 entraînable -> P3 -> RTDETRHead
```

La loss reste la loss RT-DETR one-to-one :

\[
\mathcal{L}_{RTDETR}
=
\lambda_{cls}\mathcal{L}_{varifocal}
+ \lambda_{box}\mathcal{L}_{L1}
+ \lambda_{giou}\mathcal{L}_{GIoU}
\]

avec matching Hungarian entre les `num_queries` prédictions et les objets ground truth. Les sorties auxiliaires du decoder sont aussi supervisées :

\[
\mathcal{L}_{total}
=
\mathcal{L}_{dec,L}
+ \mathcal{L}_{enc}
+ \sum_{l=1}^{L-1}\mathcal{L}_{dec,l}
\]

Cette variante teste si la tête RT-DETR a besoin d'adapter les features P3 elles-mêmes. La comparaison 2.1/2.2 sépare donc :

- l'effet de la tête RT-DETR seule ;
- l'effet d'un ré-entraînement complet backbone + tête.

### 3.4. Ablation Simple : YOLOv11NoNeck

`YOLOv11NoNeck` conserve le backbone YOLOv11n jusqu'à \(P_5\), mais supprime le neck FPN/PAN. Les trois cartes backbone sont donc envoyées directement à la tête YOLO `Detect` :

```text
image -> backbone YOLOv11n -> P3, P4, P5 -> Detect
```

La loss reste la loss YOLO standard avec `TaskAlignedAssigner`. Cette ablation teste uniquement si le neck multi-échelle YOLO apporte un gain par rapport à des têtes appliquées directement sur les features backbone.

### 3.5. No-Neck Avec Nouvelle Tête Déformable Par Échelle

Cette ablation ne teste pas un nouveau neck : elle remplace la tête de détection. Le backbone `YOLOv11NoNeck` produit directement \(P_3, P_4, P_5\), sans FPN/PAN, puis une tête one-to-one déformable prédit les boîtes en trois étapes.

```text
image -> backbone YOLOv11n no-neck -> P3, P4, P5

étape 1 : têtes spécialisées par échelle
P3 -> petits objets  -> 64 queries
P4 -> objets moyens  -> 32 queries
P5 -> grands objets  -> 16 queries

étape 2 : fusion déformable inter-échelle
queries P3/P4/P5 -> attention sur P3,P4,P5 -> boîtes sans contrainte de taille

étape 3 : raffinement final
queries fusionnées -> attention sur P3,P4,P5 -> boîtes finales
```

#### 3.5.1. Première Couche : Spécialisation Par Taille

La première couche est contrainte par niveau. Les queries de \(P_3\) détectent les petits objets, celles de \(P_4\) les objets moyens, et celles de \(P_5\) les grands objets. L'assignation des objets aux niveaux suit l'échelle d'ancre ou le stride associé au niveau.

Pour un niveau \(P_l\), la couche spécialisée applique une attention déformable mono-niveau :

\[
Q_l^{(1)} =
\operatorname{DeformAttn}(Q_l^{(0)}, P_l, b_l^{(0)}, K=16)
\]

Cette première couche sert donc à initialiser des candidates cohérentes avec l'échelle spatiale du niveau.

#### 3.5.2. Deuxième Couche : Fusion Sans Contrainte D'Échelle

Les sorties spécialisées \(Q_3^{(1)}, Q_4^{(1)}, Q_5^{(1)}\) sont ensuite concaténées. La deuxième couche prend en entrée l'ensemble des queries précédentes et peut échantillonner dans les trois cartes \(P_3, P_4, P_5\).

\[
Q^{(2)} =
\operatorname{DeformAttn}(Q^{(1)}, \{P_3,P_4,P_5\}, b^{(1)}, K=16)
\]

À partir de cette couche, il n'y a plus de contrainte petit/moyen/grand : chaque query peut corriger sa boîte et sa classe en utilisant l'information multi-échelle.

#### 3.5.3. Troisième Couche : Raffinement Final

La dernière couche répète le mécanisme de raffinement sur les queries fusionnées :

\[
Q^{(3)} =
\operatorname{DeformAttn}(Q^{(2)}, \{P_3,P_4,P_5\}, b^{(2)}, K=16)
\]

Les boîtes sont raffinées à chaque étape par :

\[
b^{(t+1)} =
\sigma\left(\Delta b^{(t)} + \sigma^{-1}(b^{(t)})\right)
\]

La sortie finale contient `64 + 32 + 16 = 112` prédictions one-to-one. Cette ablation teste donc une nouvelle tête déformable spécialisée au départ par taille d'objet, puis libérée de cette contrainte grâce à deux couches de raffinement multi-échelle.

## 4. YOLOv11TransformerNeck

`YOLOv11TransformerNeck` conserve le backbone YOLOv11 jusqu'aux cartes multi-échelles \(P_3, P_4, P_5\), puis remplace le neck FPN/PAN convolutionnel par `TransformerPyramidNeck`. La tête de détection reste une tête YOLO classique `Detect`.

Le chemin est donc :

```text
image -> backbone YOLOv11 -> P3, P4, P5 -> TransformerPyramidNeck -> Detect
```

Contrairement à `YOLOv11RTDETRHead`, cette ablation ne remplace pas la tête par une tête RT-DETR. Elle teste uniquement l'effet d'un neck transformer pour mélanger les informations entre les échelles.

### 4.1. Projection Des Cartes Multi-Échelles

Soient trois cartes issues du backbone :

\[
P_l \in \mathbb{R}^{B \times C_l \times H_l \times W_l}, \quad l \in \{3,4,5\}
\]

où :

- \(B\) est la taille du batch ;
- \(C_l\) est le nombre de canaux de l'échelle \(P_l\) ;
- \(H_l, W_l\) sont la hauteur et la largeur de l'échelle \(P_l\).

Chaque carte est projetée vers une dimension commune \(d\), égale à `transformer_d_model` :

\[
\tilde{P}_l = \phi_l(P_l), \quad \tilde{P}_l \in \mathbb{R}^{B \times d \times H_l \times W_l}
\]

où \(\phi_l\) est une convolution \(1 \times 1\). Cette projection permet de concaténer les tokens de P3, P4 et P5 dans un même espace latent.

### 4.2. Construction Des Tokens

Chaque carte projetée est aplatie spatialement :

\[
X_l = \operatorname{Flatten}(\tilde{P}_l), \quad X_l \in \mathbb{R}^{B \times N_l \times d}
\]

avec :

\[
N_l = H_l W_l
\]

Un encodage de position est ajouté :

\[
T_l = X_l + e_l(x,y) + a_l
\]

où :

- \(e_l(x,y)\) est une projection linéaire des coordonnées normalisées \((x,y)\) ;
- \(a_l \in \mathbb{R}^{d}\) est un embedding appris spécifique au niveau \(l\) ;
- \(T_l\) est la séquence de tokens enrichie pour l'échelle \(l\).

Les tokens des trois échelles sont concaténés :

\[
T = [T_3; T_4; T_5] \in \mathbb{R}^{B \times N \times d}
\]

avec :

\[
N = N_3 + N_4 + N_5
\]

À résolution \(256 \times 256\), avec des strides \((8,16,32)\), on obtient typiquement :

```text
P3 : 32 x 32 = 1024 tokens
P4 : 16 x 16 = 256 tokens
P5 :  8 x  8 = 64 tokens
N  = 1344 tokens
```

### 4.3. Mélange Global Par Self-Attention

Le neck applique un `TransformerEncoder` sur la séquence concaténée :

\[
Z = \operatorname{TransformerEncoder}(T)
\]

Il ne s'agit pas d'une cross-attention explicite entre niveaux, mais d'une self-attention globale sur tous les tokens P3/P4/P5 concaténés. Pour une tête d'attention, le mécanisme est :

\[
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)
\]

\[
Y = AV
\]

où :

- \(Q = TW_Q\), \(K = TW_K\), \(V = TW_V\) sont les queries, keys et values ;
- \(W_Q, W_K, W_V\) sont des projections apprises ;
- \(d_h = d / n_h\) est la dimension par tête ;
- \(n_h\) est le nombre de têtes `transformer_num_heads` ;
- \(A \in \mathbb{R}^{B \times n_h \times N \times N}\) est la matrice d'attention.

Le coût théorique de cette attention est quadratique en nombre de tokens :

\[
\mathcal{O}(N^2 d)
\]

Cette propriété est importante : même si le modèle peut avoir peu de paramètres, le coût réel de l'attention peut être élevé lorsque \(N\) augmente.

### 4.4. Reconstruction Des Cartes

Après le transformer, la séquence \(Z\) est découpée selon les tailles originales :

\[
Z = [Z_3; Z_4; Z_5]
\]

puis chaque \(Z_l\) est remis en carte 2D :

\[
\hat{P}_l \in \mathbb{R}^{B \times d \times H_l \times W_l}
\]

Une projection \(1 \times 1\) ramène ensuite la carte vers son nombre de canaux original :

\[
\Delta P_l = \psi_l(\hat{P}_l), \quad \Delta P_l \in \mathbb{R}^{B \times C_l \times H_l \times W_l}
\]

où \(\psi_l\) est une convolution \(1 \times 1\).

La sortie finale du neck est résiduelle :

\[
P_l^{out} = P_l + \alpha \Delta P_l
\]

où :

- \(\alpha\) est `residual_scale`, un paramètre appris initialisé par `transformer_residual_scale` ;
- \(\Delta P_l\) est la correction produite par le transformer.

Dans la configuration actuelle, `transformer_residual_scale=0.0`. Au départ :

\[
P_l^{out} = P_l
\]

Le modèle commence donc exactement depuis les features YOLOv11, puis apprend progressivement à utiliser la correction transformer si \(\alpha\) devient non nul.

### 4.5. Variante Déformable

La variante `transformer_neck_type="deformable"` remplace le `TransformerEncoder` dense par `DeformablePyramidNeck`. Les tokens P3/P4/P5 sont toujours projetés dans une dimension commune \(d\), mais chaque query n'attend qu'un petit nombre de points par niveau :

\[
\operatorname{MSDeformAttn}(q_i)
=
\sum_{l=3}^{5}
\sum_{k=1}^{K}
a_{i,l,k} \, V_l(p_{i,l} + \Delta p_{i,l,k})
\]

où :

- \(q_i\) est le token query ;
- \(K\) est `transformer_num_points` ;
- \(p_{i,l}\) est la position de référence normalisée au niveau \(l\) ;
- \(\Delta p_{i,l,k}\) est un offset appris ;
- \(a_{i,l,k}\) est le poids d'attention appris.

Le coût principal passe de \(\mathcal{O}(N^2d)\) pour l'attention dense à \(\mathcal{O}(NLKd)\), avec \(L=3\) niveaux. Cette variante conserve un mélange multi-échelle, mais évite la matrice d'attention dense entre tous les tokens.

## 5. Backbones Transformers : Swin Et DAT

Ces deux ablations remplacent le backbone convolutionnel YOLOv11 par un backbone transformer hiérarchique, en conservant à l'identique le neck FPN/PAN et la tête `Detect`. Le neck et la tête étant gelés entre les deux variantes, la comparaison Swin/DAT isole directement l'effet du type d'attention dans le backbone.

Le chemin commun est :

```text
image -> PatchEmbed (4×4, stride 4) -> stages transformer -> P3, P4, P5 -> neck YOLOv11 -> Detect
```

La configuration partagée est :

```text
embed_dim = 64    (= 256 × width_mult)
depths    = (2, 2, 4, 2)
num_heads = (2, 4, 8, 8)
mlp_ratio = 4.0
```

**`YOLOv11SwinBackbone`** utilise le Swin Transformer (Liu et al., 2021) : chaque token n'attend que ses voisins dans une fenêtre locale de taille fixe \(w \times w\), avec décalage cyclique entre couches alternées pour permettre un échange d'information inter-fenêtres. Le coût d'attention est \(\mathcal{O}(w^4 d)\) par fenêtre, indépendant de la résolution. Voir [Liu et al., arXiv:2103.14030](https://arxiv.org/abs/2103.14030).

**`YOLOv11DATBackbone`** utilise le DAT (*Deformable Attention Transformer*, Xia et al., 2022) : les blocs alternent attention locale par fenêtres (pairs) et attention déformable (impairs). Dans les blocs déformables, chaque token peut observer \(r^2\) positions adaptatives apprises sur l'ensemble de la feature map, sans contrainte de localité. Le coût est \(\mathcal{O}(HW \cdot r^2 \cdot d)\) avec \(r=7\). Voir [Xia et al., arXiv:2201.00520](https://arxiv.org/abs/2201.00520).

## 6. YOLOv11n Backbone P2-P5 Direct Heads

Cette ablation supprime le neck et expose directement les niveaux \(P_2, P_3, P_4, P_5\) du backbone YOLOv11n. Elle évite \(P_1\), trop dense en points candidats, et force \(P_2\) à porter la détection des petits objets.

### 6.1. One2Many TAL Sur P2-P5

Le premier modèle, entraîné sur `cuda:1`, applique une tête YOLO one-to-many sur \(P_2, P_3, P_4, P_5\) :

```text
image -> YOLOv11n backbone tronqué -> P2, P3, P4, P5 -> Detect(P2..P5)
```

La loss reste la loss YOLO standard avec `TaskAlignedAssigner`, classification, régression box et DFL. \(P_2\) reçoit explicitement les petits objets afin d'éviter que les niveaux plus profonds dominent l'assignation. Il n'y a pas de neck FPN/PAN : cette variante mesure la valeur brute des niveaux backbone pour une détection dense multi-échelle.

### 6.2. One2One RT-DETR Sur P2-P5

Le second modèle, entraîné sur `cuda:0`, reprend le backbone du modèle 6.1, supprime les têtes one-to-many et les remplace par des têtes one-to-one RT-DETR :

```text
image -> YOLOv11n backbone tronqué -> P2, P3, P4, P5
      -> têtes deformable attention par niveau
      -> decoder deformable final multi-level
      -> classes + boîtes finales
```

Chaque \(P_i\) est projeté vers une dimension commune, puis traité par une tête d'attention déformable propre au niveau. Le budget de queries dépend de l'échelle :

```text
P2 : 80 boîtes
P3 : 40 boîtes
P4 : 20 boîtes
P5 : 10 boîtes
```

Une dernière couche d'attention déformable agrège les sorties des têtes par niveau pour produire les boîtes finales. L'entraînement utilise un matching Hungarian one-to-one avec loss classification, L1 box et GIoU, sans `TaskAlignedAssigner` ni NMS imposé.

## 7. Coûts Des Modèles

Les coûts ci-dessous sont calculés avec `core/scripts/report_model_costs.py`. Les MACs corrigées ajoutent les opérations d'attention que `thop` ne compte pas correctement : attention spatiale YOLO, self-attention transformer, attention déformable RT-DETR et encodeur du `TransformerPyramidNeck`.

| Model | Input | Params | MACs corrigées | FLOPs corrigées | MACs attention ajoutées |
| --- | --- | ---: | ---: | ---: | ---: |
| MR_YOLO | multi-résolution | 2.44M | 2.79G | 5.57G | 17.00M |
| TF_Attn_Yolo | 1x1x256x256 | 2.37M | 498.38M | 996.75M | 1.64M |
| YOLOv11 | 1x1x256x256 | 3.64M | 631.28M | 1.26G | 1.64M |
| YOLOv11_No_Neck | 1x1x256x256 | 2.88M | 509.82M | 1.02G | 1.64M |
| YOLOv11_RTDETR_Head | 1x1x256x256 | 6.00M | 975.08M | 1.95G | 59.16M |
| YOLOv11_RTDETR_Full | 1x1x256x256 | 7.28M | 1.26G | 2.52G | 59.16M |
| YOLOv11_P3_Direct | 1x1x256x256 | 232.85K | 313.20M | 626.39M | 0 |
| YOLOv11_P3_RTDETR | 1x1x256x256 | 3.07M | 661.22M | 1.32G | 57.83M |
| YOLOv11_Transformer_Neck | 1x1x256x256 | 3.13M | 1.21G | 2.41G | 669.12M |
| YOLOv11_Deformable_Neck | 1x1x256x256 | 3.12M | 699.86M | 1.40G | 3.70M |
| YOLOv11_Swin_Backbone | 1x1x256x256 | 6.10M | 3.28G | 6.56G | 142.61M |
| YOLOv11_DAT_Backbone | 1x1x256x256 | — | — | — | — |
| YOLOv11_P2P5_One2Many | 1x1x256x256 | 2.28M | 1.14G | 2.28G | 0 |
| YOLOv11_P2P5_RTDETR_One2One | 1x1x256x256 | 10.28M | 1.23G | 2.46G | 6.14M |

Pour `YOLOv11_P2P5_RTDETR_One2One`, la mesure utilise les budgets \(80/40/20/10\), 6 couches d'attention déformable par niveau et 1 couche finale déformable multi-niveau.


## 8. Sources

- DEYO, implémentation officielle : https://github.com/ouyanghaodong/DEYO  
  Référence utilisée pour l'architecture `RTDETRDecoder`, le transfert depuis un modèle YOLO, le choix `nc` sans classe `no-object`, la varifocal loss, le decoder à raffinement itératif et la configuration d'entraînement.

- Ouyang et al., *DEYO: DETR with YOLO for End-to-End Object Detection* : https://arxiv.org/abs/2402.16370  
  Référence de l'ablation YOLO + DETR/RT-DETR et de la motivation consistant à obtenir une détection end-to-end sans NMS.

- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* / RT-DETR : https://arxiv.org/abs/2304.08069  
  Référence utilisée pour l'architecture RT-DETR, la sélection de queries guidée par les scores encodeur et l'objectif de détection end-to-end temps réel.

- Carion et al., *End-to-End Object Detection with Transformers* / DETR : https://arxiv.org/abs/2005.12872  
  Référence utilisée pour le principe de prédiction en ensemble, le matching biparti Hungarian et la loss one-to-one.

- Zhu et al., *Deformable DETR: Deformable Transformers for End-to-End Object Detection* : https://arxiv.org/abs/2010.04159  
  Référence utilisée pour l'attention déformable multi-échelle, où l'attention se limite à un petit nombre de points échantillonnés autour de références spatiales.

- Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* : https://arxiv.org/abs/2103.14030  
  Référence originale du backbone Swin, du patch embedding hiérarchique et de l'attention locale par fenêtres décalées.

- Xia et al., *Vision Transformer with Deformable Attention* / DAT : https://arxiv.org/abs/2201.00520  
  Référence pour l'attention déformable dans un backbone hiérarchique : génération d'offsets par réseau léger, grille de référence uniforme, bornage tanh, et alternance blocs locaux / déformables pour stabiliser l'entraînement.
