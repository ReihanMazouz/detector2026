# Transformer multi-résolution pour spectrogrammes temps-fréquence

## Motivation

Les backbones multi-résolution actuels reposent principalement sur des branches convolutives indépendantes, suivies d'une fusion par concaténation, somme pondérée ou attention spatiale/canal. Cette stratégie est efficace pour agréger des cartes de caractéristiques ramenées sur une grille commune, mais elle impose une fusion relativement locale et souvent implicite entre résolutions.

L'objectif proposé ici est d'introduire un mécanisme de type Transformer permettant une interaction explicite entre plusieurs représentations temps-fréquence du même signal, calculées avec des résolutions différentes. L'idée centrale est qu'un événement acoustique, électrique ou physique peut être plus lisible dans une résolution courte, qui localise mieux le temps, ou dans une résolution longue, qui localise mieux la fréquence. La cross-attention inter-résolution doit permettre au modèle d'exploiter ces complémentarités.

## Cadre formel

On suppose que le même signal temporel est représenté par un ensemble de spectrogrammes :

```text
X = {X^(1), X^(2), ..., X^(R)}
```

où `R` désigne le nombre de résolutions disponibles. Pour une résolution `r`, le spectrogramme est noté :

```text
X^(r) ∈ R^{C_r x T_r x F_r}
```

avec `C_r` canaux, `T_r` pas temporels et `F_r` bandes fréquentielles. Chaque résolution peut provenir d'une transformée temps-fréquence différente, par exemple une STFT avec une taille de fenêtre, un recouvrement ou un nombre de points FFT spécifiques.

Une cellule temps-fréquence `i = (t, f)` d'une résolution `r` ne doit pas être interprétée uniquement comme un indice discret. Elle correspond à un support physique :

```math
\Omega_i^{(r)}
= [\tau_i^{(r)} - \Delta \tau_r / 2,\ \tau_i^{(r)} + \Delta \tau_r / 2]
\times
[\nu_i^{(r)} - \Delta \nu_r / 2,\ \nu_i^{(r)} + \Delta \nu_r / 2],
```

où `\tau_i^{(r)}` et `\nu_i^{(r)}` sont respectivement le centre temporel et le centre fréquentiel de la cellule, tandis que `\Delta \tau_r` et `\Delta \nu_r` décrivent les résolutions temporelle et fréquentielle associées.

Cette formulation est importante : deux cellules ayant les mêmes indices `(t, f)` dans deux résolutions différentes ne correspondent pas nécessairement aux mêmes zones physiques du plan temps-fréquence.

## Principe général

L'architecture proposée suit quatre étapes :

1. encoder chaque spectrogramme avec une branche spécifique à sa résolution ;
2. convertir les cartes temps-fréquence en tokens enrichis par des coordonnées physiques ;
3. faire communiquer les résolutions par cross-attention géométriquement contrainte ;
4. fusionner les représentations sur une grille cible ou via des requêtes de détection.

Le schéma conceptuel est le suivant :

```text
Spectrogrammes multi-résolution
        ↓
Branches d'encodage par résolution
        ↓
Tokens temps-fréquence avec coordonnées physiques
        ↓
Self-attention intra-résolution
        ↓
Cross-attention inter-résolution
        ↓
Fusion multi-échelle
        ↓
Tête de détection ou classification
```

## Encodage des tokens

Pour chaque résolution `r`, une branche convolutionnelle légère ou un stem de type `BranchBackbone` transforme le spectrogramme initial en une carte de caractéristiques :

```math
H^{(r)} = B_r(X^{(r)}),
```

où :

```math
H^{(r)} \in \mathbb{R}^{D_r \times T'_r \times F'_r}.
```

Chaque position de cette carte est ensuite projetée dans un espace latent commun de dimension `d` :

```math
z_i^{(r)} = W_r h_i^{(r)} + p_i^{(r)}.
```

Le terme `W_r` est une projection linéaire spécifique à la résolution. Le terme `p_i^{(r)}` est un encodage positionnel contenant les informations physiques utiles :

```math
p_i^{(r)} = \phi(
    \tau_i^{(r)},
    \nu_i^{(r)},
    \Delta \tau_r,
    \Delta \nu_r,
    r
).
```

La fonction `\phi` peut être un encodage sinusoïdal, un petit MLP, ou une combinaison des deux. Contrairement à un encodage positionnel classique, elle n'encode pas seulement la position discrète dans la matrice, mais aussi la signification physique de chaque token.

## Self-attention intra-résolution

Avant de mélanger les résolutions, chaque branche doit apprendre les structures propres à son spectrogramme. On applique donc des blocs de self-attention intra-résolution :

```math
\bar{Z}^{(r)} = \mathrm{SelfAttn}_r(Z^{(r)}).
```

Dans une implémentation pratique, il est préférable de ne pas utiliser une attention globale complète sur toute la carte temps-fréquence. Les alternatives recommandées sont :

- attention locale dans des fenêtres temps-fréquence ;
- attention axiale, d'abord selon l'axe temporel puis selon l'axe fréquentiel ;
- attention de type Swin Transformer avec fenêtres décalées ;
- blocs hybrides combinant convolutions séparables temps-fréquence et attention locale.

Cette étape permet d'extraire les motifs internes à chaque résolution : transitoires, bandes fréquentielles, structures harmoniques, textures locales ou signatures spectrales compactes.

## Cross-attention inter-résolution

Le cœur de la proposition est un bloc de cross-attention permettant à une résolution `r` d'interroger les autres résolutions `s != r`.

Pour un token `i` de la résolution `r`, les requêtes, clés et valeurs sont définies par :

```math
q_i^{(r)} = W_Q^{(r)} z_i^{(r)},
```

```math
k_j^{(s)} = W_K^{(s)} z_j^{(s)}, \quad
v_j^{(s)} = W_V^{(s)} z_j^{(s)}.
```

La mise à jour inter-résolution peut alors s'écrire :

```math
\tilde{z}_i^{(r)}
= z_i^{(r)}
+ \gamma_r
\sum_{s \neq r}
\sum_{j \in \mathcal{N}_{r \leftarrow s}(i)}
\alpha_{ij}^{(r \leftarrow s)} v_j^{(s)}.
```

Le coefficient `\gamma_r` est un paramètre de gating appris, idéalement initialisé proche de zéro. Il permet de stabiliser l'entraînement et d'éviter que les branches ne se mélangent trop fortement dès les premières itérations.

Les poids d'attention sont donnés par :

```math
\alpha_{ij}^{(r \leftarrow s)}
=
\mathrm{softmax}_{j}
\left(
\frac{
\langle q_i^{(r)}, k_j^{(s)} \rangle
}{\sqrt{d}}
+ b_{ij}^{(r,s)}
+ m_{ij}^{(r,s)}
\right).
```

Le terme `b_{ij}^{(r,s)}` est un biais géométrique appris, dépendant des relations physiques entre les deux tokens :

```math
b_{ij}^{(r,s)}
= g(
\tau_i^{(r)} - \tau_j^{(s)},
\nu_i^{(r)} - \nu_j^{(s)},
\mathrm{IoU}_t(\Omega_i^{(r)}, \Omega_j^{(s)}),
\mathrm{IoU}_f(\Omega_i^{(r)}, \Omega_j^{(s)}),
r,
s
).
```

Le terme `m_{ij}^{(r,s)}` est un masque d'attention. Il vaut `0` si le token `j` est autorisé comme voisin du token `i`, et `-∞` sinon. Ce masque permet d'éviter une attention dense entre tous les points de toutes les résolutions.

## Voisinage géométrique

Une attention entièrement dense entre toutes les résolutions aurait un coût quadratique :

```math
\mathcal{O}\left(
\left(\sum_{r=1}^{R} T'_r F'_r\right)^2
\right).
```

Ce coût est souvent prohibitif pour des spectrogrammes de grande taille. Il est donc recommandé d'utiliser une cross-attention restreinte à un voisinage temps-fréquence :

```math
\mathcal{N}_{r \leftarrow s}(i)
=
\left\{
j :
|\tau_i^{(r)} - \tau_j^{(s)}| \leq \kappa_t \max(\Delta \tau_r, \Delta \tau_s),
\ 
|\nu_i^{(r)} - \nu_j^{(s)}| \leq \kappa_f \max(\Delta \nu_r, \Delta \nu_s)
\right\}.
```

Les constantes `\kappa_t` et `\kappa_f` contrôlent la largeur du contexte accessible dans les axes temporel et fréquentiel. Ce voisinage peut aussi être défini par recouvrement des supports `\Omega_i^{(r)}` et `\Omega_j^{(s)}`.

Cette restriction a un double intérêt :

- elle réduit fortement le coût mémoire ;
- elle introduit un biais inductif cohérent avec la géométrie du plan temps-fréquence.

## Bloc Transformer proposé

Un bloc élémentaire peut être défini comme suit, pour chaque résolution `r` :

```text
Z_r ← Z_r + LocalSelfAttention(LayerNorm(Z_r))
Z_r ← Z_r + γ_r CrossResolutionAttention(LayerNorm(Z_r), {LayerNorm(Z_s), s ≠ r})
Z_r ← Z_r + MLP(LayerNorm(Z_r))
```

En notation compacte :

```math
Z_l^{(r)}
=
\mathrm{MRBlock}_l
\left(
Z_{l-1}^{(r)},
\{Z_{l-1}^{(s)}\}_{s \neq r}
\right).
```

Plusieurs blocs peuvent être empilés :

```math
\{Z_L^{(r)}\}_{r=1}^{R}
=
\mathrm{MRTransformer}
\left(
\{Z_0^{(r)}\}_{r=1}^{R}
\right).
```

L'architecture reste modulaire : les branches convolutionnelles existantes peuvent produire les premiers niveaux de représentation, puis les blocs Transformer assurent la communication inter-résolution.

## Fusion finale

Deux stratégies de sortie sont envisageables.

### Fusion sur une grille cible

On choisit une résolution principale `r*`, par exemple celle qui correspond le mieux à la tête de détection actuelle. Les autres résolutions enrichissent cette grille par cross-attention :

```math
Z_{\mathrm{out}}
=
\mathrm{CrossAttn}
\left(
Z^{(r*)},
\{Z^{(r)}\}_{r=1}^{R}
\right).
```

La sortie est ensuite remise sous forme de carte de caractéristiques puis transmise à une tête de détection pyramidale, par exemple sous la forme `P3`, `P4`, `P5`.

Cette option est la plus compatible avec les backbones existants.

### Fusion par requêtes de détection

Une autre possibilité consiste à introduire un ensemble de requêtes apprises, analogues aux object queries de DETR :

```math
Q_{\mathrm{det}} \in \mathbb{R}^{N_q \times d}.
```

Ces requêtes interrogent l'ensemble des tokens multi-résolution :

```math
Y
=
\mathrm{CrossAttn}
\left(
Q_{\mathrm{det}},
\mathrm{concat}(Z^{(1)}, ..., Z^{(R)})
\right).
```

Cette formulation est particulièrement intéressante si la tâche consiste à détecter un nombre variable d'événements localisés dans le plan temps-fréquence.

## Architecture recommandée

La version recommandée pour une première implémentation robuste est la suivante :

```text
Entrées :
  X^(1), ..., X^(R)

Pour chaque résolution r :
  BranchBackbone ou stem convolutionnel léger
  Projection vers une dimension commune d
  Ajout d'un encodage positionnel physique

Répéter L fois :
  Self-attention locale intra-résolution
  Cross-attention inter-résolution contrainte par la géométrie temps-fréquence
  MLP
  Connexions résiduelles et LayerNorm

Sortie :
  Fusion sur une grille cible
  Construction éventuelle de P3/P4/P5
  Tête de détection existante
```

Cette proposition conserve les avantages des branches spécialisées par résolution tout en ajoutant une communication explicite entre elles. La résolution courte peut apporter une localisation temporelle fine, tandis que la résolution longue peut apporter une localisation fréquentielle plus précise. La cross-attention apprend à pondérer dynamiquement ces informations selon le contenu du signal.

## Points d'attention pour l'implémentation

1. Les encodages positionnels doivent être exprimés en coordonnées physiques, pas uniquement en indices de matrice.
2. La cross-attention dense doit être évitée pour les grandes cartes temps-fréquence.
3. Le masque de voisinage doit respecter les différences de pas temporels et fréquentiels entre résolutions.
4. Les branches doivent conserver une identité propre ; il est donc utile d'ajouter un embedding de résolution.
5. Un gating appris sur la cross-attention stabilise l'entraînement.
6. Une fusion finale sur une grille cible est probablement le chemin le plus simple pour rester compatible avec l'architecture actuelle.
7. Les coûts mémoire doivent être mesurés dès les premiers prototypes, car le nombre de tokens peut croître rapidement.

## Variante minimale pour prototype

Une première version expérimentale peut être volontairement simple :

1. utiliser les sorties `P3` de chaque branche existante ;
2. projeter chaque `P3` vers une dimension commune `d` ;
3. ajouter un encodage de résolution et un encodage 2D temps-fréquence ;
4. appliquer un petit nombre de blocs de cross-attention ;
5. remettre la résolution cible sous forme de carte 2D ;
6. continuer avec la pyramide `P3/P4/P5` existante.

Cette variante permet de tester rapidement l'intérêt de la communication inter-résolution sans réécrire entièrement le backbone.

## Hypothèse scientifique

L'hypothèse principale est que les spectrogrammes multi-résolution contiennent des informations complémentaires mais non parfaitement alignées sur une grille commune. Une fusion par concaténation impose au modèle de résoudre implicitement cet alignement dans les couches suivantes. Une cross-attention inter-résolution, guidée par les coordonnées physiques des tokens, fournit au contraire un mécanisme explicite pour associer les événements observés à différentes échelles temps-fréquence.

Ainsi, pour un événement localisé autour de `(τ, ν)`, le modèle peut apprendre à combiner :

- une résolution courte, plus fiable pour préciser `τ` ;
- une résolution longue, plus fiable pour préciser `ν` ;
- une résolution intermédiaire, utile pour stabiliser la représentation.

Cette architecture devrait donc être particulièrement adaptée aux signaux dans lesquels les événements d'intérêt possèdent simultanément des signatures temporelles brèves et des signatures fréquentielles structurées.
