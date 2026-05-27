# Ablations MR-YOLO par cross-attention inter-resolution

## Objectif

Ce package regroupe des ablations de `MR_YOLO` dont le but est d'isoler l'apport de la cross-attention inter-resolution. Contrairement a un Transformer complet, ces variantes n'utilisent pas de self-attention intra-resolution : chaque resolution conserve uniquement son encodage convolutionnel local, puis les resolutions communiquent au moment de la fusion.

On considere une liste de spectrogrammes representant le meme signal :

```text
X = {X^(1), ..., X^(R)}
```

avec :

```math
X^{(r)} \in \mathbb{R}^{B \times C_r \times H_r \times W_r}.
```

La fusion cherche a produire une representation unique :

```math
Y \in \mathbb{R}^{B \times C_{\mathrm{out}} \times H_* \times W_*},
```

ou `(H_*, W_*)` correspond a la resolution centrale, definie par defaut par `R // 2`.

## Module commun de fusion

Le module `InterResolutionCrossAttentionFusion` recoit une liste de cartes temps-frequence et applique une cross-attention ou les requetes proviennent de la resolution centrale. Les cles et valeurs proviennent des `R` resolutions.

Chaque carte est d'abord projetee dans un espace latent commun de dimension `d` :

```math
Z^{(r)} = \mathrm{Proj}_r(X^{(r)}).
```

Un encodage positionnel 2D normalise et un embedding de resolution sont ajoutes aux tokens :

```math
\hat{z}_{i}^{(r)}
= z_i^{(r)}
+ \phi(x_i, y_i)
+ e_r.
```

La sortie de fusion par defaut est calculee par attention deformable multi-resolution. Pour chaque query \(q_i\) issue de la grille centrale, le module echantillonne \(K\) points dans chaque resolution :

```math
\mathrm{DefAttn}(q_i)
=
\sum_{r=1}^{R}
\sum_{k=1}^{K}
\alpha_{i,r,k}
\, v^{(r)}\left(p_i + \Delta p_{i,r,k}\right).
```

ou :

- \(p_i\) est la position normalisee du token central \(i\) ;
- \(\Delta p_{i,r,k}\) est un offset appris pour le point \(k\) de la resolution \(r\) ;
- \(\alpha_{i,r,k}\) est le poids d'attention appris ;
- \(v^{(r)}(\cdot)\) est la valeur echantillonnee par interpolation bilineaire dans la resolution \(r\) ;
- \(K\) correspond a `fusion_num_points`.

Cette formulation evite de redimensionner brutalement les spectrogrammes vers la grille centrale. Les resolutions gardent leurs grilles propres, et le module apprend ou echantillonner l'information utile.

Deux modes existent dans le code.

### Mode `global`

Chaque token de la grille cible peut interroger tous les tokens de toutes les resolutions :

```math
Q \in \mathbb{R}^{B \times H_*W_* \times d},
```

```math
K,V \in \mathbb{R}^{B \times \left(\sum_r H_rW_r\right) \times d}.
```

Ce mode est le plus expressif, mais son cout memoire croit avec le nombre total de tokens.

### Mode `deformable`

Chaque token de la grille centrale interroge un petit nombre de points dans chaque resolution :

```math
Q \in \mathbb{R}^{B \times H_*W_* \times d},
```

```math
V \in \mathbb{R}^{B \times \left(\sum_r H_rW_r\right) \times d}.
```

Le nombre de paires effectivement echantillonnees est proportionnel a \(H_*W_*RK\), et non a \(H_*W_* \sum_r H_rW_r\). C'est le mode par defaut des ablations.

## Architecture 1 : branches puis cross-attention

`MRYOLOBranchCrossAttentionAblation` conserve l'esprit de `MR_YOLO`.

Chaque spectrogramme \(X^{(r)}\) est d'abord traite par une branche convolutionnelle propre a sa resolution :

```math
F^{(r)} = B_r\left(X^{(r)}\right),
\qquad
F^{(r)} \in \mathbb{R}^{B \times C_r' \times H_r' \times W_r'}.
```

ou :

- \(B_r\) est le `BranchBackbone` associe a la resolution \(r\) ;
- \(F^{(r)}\) est la carte profonde produite par cette branche ;
- \(C_r'\), \(H_r'\), \(W_r'\) sont les canaux, hauteur et largeur apres encodage local.

Les cartes profondes des \(R\) branches sont ensuite fusionnees par cross-attention inter-resolution :

```math
P_3
=
\mathcal{F}_{\mathrm{cross}}
\left(
F^{(1)}, \ldots, F^{(R)}
\right),
\qquad
P_3 \in \mathbb{R}^{B \times C_3 \times H_* \times W_*}.
```

ou :

- \(\mathcal{F}_{\mathrm{cross}}\) designe `InterResolutionCrossAttentionFusion` ;
- \(P_3\) est la premiere carte pyramidale apres fusion ;
- \(C_3\) est le nombre de canaux de sortie de la fusion ;
- \((H_*, W_*)\) est la grille de la resolution centrale.

Les niveaux plus profonds sont construits par descente convolutionnelle :

```math
P_4 = \psi_4(P_3),
\qquad
P_5 = \psi_5(P_4),
```

avec :

- \(\psi_4\) une convolution stride 2 suivie d'un bloc `TFSepBlock` ;
- \(\psi_5\) une convolution stride 2 suivie de `TFSepBlock`, `SPPF` et `C2PSA`.

Le neck FPN/PAN reconstruit ensuite les cartes de detection :

```math
(\tilde{P}_3, \tilde{P}_4, \tilde{P}_5)
=
\mathcal{N}_{\mathrm{FPN/PAN}}(P_3, P_4, P_5).
```

La tete `Detect` predit enfin les distributions de boites et les logits de classes :

```math
(\hat{D}, \hat{S})
=
\mathcal{D}(\tilde{P}_3, \tilde{P}_4, \tilde{P}_5),
```

ou \(\hat{D}\) designe les distributions de regression DFL et \(\hat{S}\) les scores de classes.

Cette ablation remplace donc principalement la fusion par concatenation ou attention convolutionnelle par une fusion explicite entre resolutions.

## Architecture 2 : cross-attention sur les spectres d'entree

`MRYOLOInputCrossAttentionAblation` fusionne directement les spectres d'entree.

Chaque spectrogramme brut est d'abord encode par un stem convolutionnel leger :

```math
E^{(r)} = g_r\left(X^{(r)}\right),
\qquad
E^{(r)} \in \mathbb{R}^{B \times C_e \times H_r \times W_r}.
```

ou :

- \(g_r\) est une convolution \(3 \times 3\) propre a la resolution \(r\) ;
- \(C_e\) est le nombre de canaux intermediaires `encoder_channels` ;
- la resolution spatiale \((H_r, W_r)\) est conservee a ce stade.

Les cartes encodees sont fusionnees avant tout backbone profond :

```math
X_{\mathrm{fused}}
=
\mathcal{F}_{\mathrm{cross}}
\left(
E^{(1)}, \ldots, E^{(R)}
\right),
\qquad
X_{\mathrm{fused}}
\in
\mathbb{R}^{B \times C_{\mathrm{in}} \times H_* \times W_*}.
```

ou :

- \(X_{\mathrm{fused}}\) est un spectrogramme latent unique ;
- \(C_{\mathrm{in}}\) est le nombre de canaux attendu par YOLOv11 ;
- \((H_*, W_*)\) est la taille de la resolution centrale.

Ce tenseur fusionne remplace l'image d'entree d'un YOLOv11 classique :

```math
(P_3, P_4, P_5)
=
\mathcal{B}_{\mathrm{YOLOv11}}
\left(
X_{\mathrm{fused}}
\right),
```

```math
(\tilde{P}_3, \tilde{P}_4, \tilde{P}_5)
=
\mathcal{N}_{\mathrm{YOLOv11}}
(P_3, P_4, P_5),
```

```math
(\hat{D}, \hat{S})
=
\mathcal{D}
(\tilde{P}_3, \tilde{P}_4, \tilde{P}_5).
```

ou :

- \(\mathcal{B}_{\mathrm{YOLOv11}}\) est le backbone YOLOv11 ;
- \(\mathcal{N}_{\mathrm{YOLOv11}}\) est le neck YOLOv11 ;
- \(\mathcal{D}\) est la tete `Detect`.

Cette variante teste si la communication entre resolutions doit intervenir avant tout backbone profond.

Par defaut, cette architecture doit etre instanciee en mode `deformable`.
Le mode `global` reste disponible dans la classe pour des tests controles, mais il ne doit pas etre inclus dans les scripts de baseline : son cout memoire est prohibitif sur les spectres d'entree haute resolution.

## Architecture 3 : patch spatial attention multi-resolution

`MRPatchSpatialAttentionBlock` est l'ablation transformer retenue pour la suite. Les patches ne remplacent pas le backbone et ne produisent pas une pyramide. Ils servent a calculer une carte d'attention spatiale, appliquee ensuite sur les pixels/features d'entree du bloc.

Le bloc prend une liste de feature maps multi-resolution :

```math
F^{(r)} \in \mathbb{R}^{B \times C_r \times T_r \times F_r}.
```

### Patching latent aligne

Chaque feature map est patchifiee avec une taille anisotrope propre a sa resolution :

```math
Z^{(r)} = \mathrm{PatchEmbed}_r(F^{(r)}),
\qquad
Z^{(r)} \in \mathbb{R}^{B \times N \times d}.
```

Les tailles de patch sont choisies pour que toutes les resolutions produisent la meme grille latente :

```math
\frac{T_r}{p_{t,r}} = H_*,
\qquad
\frac{F_r}{p_{f,r}} = W_*.
```

Chaque token recoit un embedding de position partage et un embedding de resolution.

### Attention intra/inter-resolution

Le bloc applique d'abord une attention deformable intra-resolution :

```math
\tilde{Z}^{(r)} = \mathrm{DefSelfAttn}(Z^{(r)}).
```

Puis, pour chaque position latente `p`, il applique une attention uniquement entre les resolutions correspondantes :

```math
\bar{S}(p) =
\mathrm{MHSA}_{\mathrm{res}}
\left(
[\tilde{Z}^{(1)}(p), \ldots, \tilde{Z}^{(R)}(p)]
\right).
```

Il n'y a pas d'attention globale entre toutes les positions et toutes les resolutions.

### Attention par pixel

Les tokens mis a jour ne sont pas reconstruits en feature map dense. Ils produisent une carte d'attention spatiale basse resolution :

```math
a^{(r)} = \mathrm{Linear}(\bar{Z}^{(r)}),
\qquad
a^{(r)} \in \mathbb{R}^{B \times 1 \times H_* \times W_*}.
```

Cette carte est interpolee vers la taille de la feature map d'entree :

```math
A^{(r)} = \mathrm{Upsample}(a^{(r)}, T_r, F_r).
```

La sortie est une modulation residuelle pixel-wise :

```math
F_{out}^{(r)} =
F^{(r)} + \alpha_r \left(2\sigma(A^{(r)}) - 1\right) \odot F^{(r)}.
```

`alpha_r` est appris et initialise a zero afin que le bloc commence comme une identite.

Le bloc teste donc une hypothese ciblee : est-ce qu'une attention latente multi-resolution, calculee sur patches alignes, peut recalibrer spatialement les features convolutionnelles ?

## Complexite des ablations

Les couts ci-dessous sont calcules pour une entree batch 1 avec les 5 resolutions :

```text
64x1024, 128x512, 256x256, 512x128, 1024x64
```

Configuration commune :

- famille `n` : `width_mult=0.25` ;
- `num_classes=20`, `reg_max=16`, `in_ch=1` ;
- fusion cross-attention : `fusion_d_model=128`, `fusion_num_heads=4`, `fusion_num_layers=1`, `fusion_num_points=4`, `fusion_ffn_ratio=2.0`, `fusion_dropout=0.0` ;
- les MACs corrigees ajoutent les operations d'attention non comptees directement par `thop` : `MultiheadAttention`, attention deformable et attention spatiale `C2PSA`.

| Modele | Details architecture | Params | MACs corrigees | FLOPs corrigees | MACs attention ajoutes |
| --- | --- | ---: | ---: | ---: | ---: |
| `MR_YOLO baseline` | `n`, 5 resolutions, `backbone_mode=TFSep_pyramid`, `outfusion_channels_mult=1` | 2.44M | 2.79G | 5.57G | 17.00M |
| `MRYOLOBranchCrossAttentionAblation`, `global` | `n`, `BranchBackbone` par resolution, fusion apres branches, `d=128`, 4 tetes, 1 couche, attention globale dense | 4.94M | 29.51G | 59.02G | 23.62G |
| `MRYOLOBranchCrossAttentionAblation`, `deformable` | `n`, `BranchBackbone` par resolution, fusion apres branches, `d=128`, 4 tetes, 1 couche, 4 points par resolution | 4.94M | 6.42G | 12.83G | 10.49M |
| `MRYOLOInputCrossAttentionAblation`, `global` | fusion avant backbone, encodeur entree `Conv`, `encoder_channels=16`, puis YOLOv11n, `d=128`, 4 tetes, 1 couche, attention globale dense | 3.78M | 5.87T | 11.74T | 5.85T |
| `MRYOLOInputCrossAttentionAblation`, `deformable` | fusion avant backbone, encodeur entree `Conv`, `encoder_channels=16`, puis YOLOv11n, `d=128`, 4 tetes, 1 couche, 4 points par resolution | 3.78M | 14.63G | 29.27G | 169.41M |
| `MR_YOLO + MRPatchSpatialAttentionBlock x2` | MR-YOLO complet avec deux blocs de recalibration spatiale appliques entre `P1->P2` et `P2->P3`, grille latente `16x16`, `d=128`, 4 tetes, 1 couche, 16 points deformables | 7.34M | 6.87G | 13.74G | 352.99M |

Lecture du tableau :

La difference principale vient du nombre de tokens traites par l'attention. Pour une attention globale, le cout dominant est :

```math
\mathcal{O}(N_q N_m d),
```

ou \(N_q\) est le nombre de queries de la resolution centrale, \(N_m\) le nombre total de tokens memoire sur toutes les resolutions, et \(d\) la dimension latente. Pour une attention deformable, le cout dominant devient :

```math
\mathcal{O}(N_q R K d),
```

ou \(R\) est le nombre de resolutions et \(K\) le nombre de points echantillonnes par resolution.

Pour `MRPatchSpatialAttentionBlock`, le cout d'attention est la somme de deux termes dans la grille latente :

```math
\mathcal{O}
\left(
R N K d
\right)
+
\mathcal{O}
\left(
N R^2 d
\right).
```

Le premier terme correspond a l'attention deformable intra-resolution. Le second correspond a l'attention inter-resolution alignee, appliquee uniquement entre les \(R\) tokens representant la meme position physique. La ligne `MR_YOLO + MRPatchSpatialAttentionBlock x2` mesure le modele MR-YOLO complet augmente avec deux insertions : une sur les features `P1` des branches, puis une sur les features `P2`. Les features mesurees sont :

```text
P1 : (64x256), (128x256), (128x128), (256x128), (256x64)
P2 : (64x128), (128x128), (64x64),  (128x128), (128x64)
```

La mesure complete combine les MACs `thop` (`6.52G`) et les MACs d'attention non comptees automatiquement (`352.99M`). Le cout des deux blocs seuls est `1.35G` MACs corrigees ; le modele complet avec branches, fusion, P4/P5, neck et tete vaut `6.87G` MACs corrigees.

- `BranchCrossAttention global` sert de reference haute : l'attention est appliquee apres les `BranchBackbone`, donc les cartes sont deja reduites, mais chaque token central regarde encore tous les tokens issus des cinq branches. Le cout monte a `29.51G` MACs, principalement a cause des `23.62G` MACs d'attention dense.
- `BranchCrossAttention deformable` garde les memes parametres que le mode global, mais remplace la matrice dense par \(R \times K = 5 \times 4\) points par query. Le cout descend a `6.42G` MACs : l'essentiel du cout vient alors du backbone/neck convolutionnel, plus de l'attention.
- `InputCrossAttention global` est pratiquement prohibitif : sur les spectres d'entree, la grille centrale contient \(65\,536\) queries et les cinq resolutions fournissent \(327\,680\) tokens memoire. Le produit \(N_qN_m\) explique les `5.85T` MACs d'attention ajoutees.
- `InputCrossAttention deformable` reste plus couteux que la fusion apres branches, car il travaille avant reduction profonde de resolution. En revanche, il evite le resize destructif et limite la fusion a \(65\,536 \times 5 \times 4\) echantillons, ce qui ramene le cout a `14.63G` MACs au lieu de plusieurs tera-MACs.
- `MR_YOLO + MRPatchSpatialAttentionBlock x2` est logiquement plus couteux que `MR_YOLO baseline`, car il ajoute deux blocs d'attention latente au modele complet. Le surcout reste nettement inferieur a la variante `patch/unpatch` a trois niveaux, car les tokens ne reconstruisent pas les cartes denses : ils produisent une carte d'attention spatiale partagee sur les canaux.
