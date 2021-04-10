# Modèles

- [Problème](#problem)
- [Modèles](#modeles)
    * [Mesure de Performance](#mesure-de-performance)
    * [Word2Vec](#word2vec)
    * [Régression Logistique](#regression-logistique)
    * [Les plus proches voisins (KNN)](#les-plus-proches-voisins-knn)
    * [Naive Bayes et n-grammes](#naive-bayes-et-n-grammes)

## Problème

Nous sommes face à un problème de classification (la classe est le world port index, e.g. FRBES pour Brest) dans le domaine plus large du NLP (Natural Language Processing).

Tous les modèles d'apprentissage prennent uniquement des données numériques en entrée, il est donc nécessaire d'encoder nos entrées, i.e définir une représentation numérique d'une chaîne de caractères.

Les méthodes utilisés sont fondés sur le même principe de vocabulaire. Un vocabulaire est une manière d'associer une chaîne de caractère à un entier (un indice), en général de manière incrémentale. Les 2 méthodes sont:
- à chaque mot on associe un entier (ex: Brest <-> 1, France <-> 2, etc)
- à chaque lettre on associe un entier (ex: a <-> 1, b <-> 2, etc)


## Modèles
Pour chacun des modèles implémentés, nous verrons l'optimisation des paramètres, la performance etc

#### Mesure de Performance

Notre métrique de précision pour évaluer les différents modèles est le **Recall @k**.
Nos modèles retournent la liste des classes possibles, avec pour chaque une valeur qui correspond à la probabilité que cette classe soit la bonne:

$${
    FRBES: 0.1,
    NLRTM: 0.2,
    FRBOD: 0.7
}$$

Le **Recall @1** consiste à regarder si la classe avec la plus grande valeur correspond à la bonne classe et sommer les bonnes réponses:

$$ \frac{nombre \, de \, correctes}{nombre \, de \, données} $$

Le **Recall @2** consiste à regarder si la bonne classe est parmi les 2 classes avec les plus grandes valeurs.

Enfin, le **recall @3** consiste à regarder si la bonne classe est parmi les 3 plus grandes valeurs.

#### Word2Vec

Ici nous utilisons un encodage avec un vocabulaire construit avec une liste de mots rencontrés dans le jeux de données d'entrainement.


##### Résultats

##### Augmentation des données: KeyAug

Nous voyons que le modèle n'a aucune faculté de généralisation sur des mots non rencontrés au temps d'entrainement.
Nous nous sommes dit qu'en incorporant des fautes d'orthographes nous pourrions rendre le modèle plus robuste, en espérant qu'il apprenne à gérer ces fautes (ex: `ST PETERSBOURG == ST PETERSBURG`)


Finalement, on voit que l'augmentation des données à un effet desastreux. Le modèle ne parvient plus à classer même les mots vu à l'entrainement.

##### Bilan

Nous sommes passé à côté d'une dimension du problème: l'absence de vocabulaire. Une part du challenge est dans l'absence d'un vocabulaire, soit à cause d'abréviation non référencées soit à cause de fautes d'orthographes.
Nos prochaines méthodes feront donc usage d'un vocabulaire constitué de caractères et non de mots.

#### Régression Logistique

##### Résultats

##### Avantages

##### Inconvénients

#### Les plus proches voisins (KNN)

##### Algorithme
- Entrainement: sauvegarder le jeux de données d'entrainement
- Prédiction: trouver les k plus proches voisins (parmi les données labellisées sauvegardées) de l'entrée. Nous attribuons la classe la plus présente parmi les voisins.

##### Résultats

##### Avantages

##### Inconvénients


#### Naive Bayes et n-grammes

Un n-gram est simplement une manière de glisser une fenêtre de taille fixée pour extraire une sous-chaîne de caractères:

![](ngrams_char.jpg)

La méthode générative de Bayes consiste à tirer parti de la formule de Bayes:

$$
\mathbb P(Y=y | X=x) = 
\frac{\mathbb P(X = x | Y= y) \mathbb P(Y=y)}{\mathbb P(X=x)}
$$
où $X$ est un n-gramme et $Y$ la classe.

À l'entrainement, le modèle apprend les termes $\mathbb P(X = x | Y= y)$ et $\mathbb P(Y=y)$:

$$
\mathbb P(X = x | Y= y) = \frac{nb \, d'occurence \, de \, x \, dans \, y}{nb \, d'apparition \, de \, y}
$$
$$
\mathbb P(Y= y) = \frac{nb \, d'apparation \, de \, y}{nb \, de \, données}
$$

Ainsi, pour prédire une classe, il suffit de réutiliser les probabilités estimées de tous les n-grammes:

$$
\mathbb P(Y=y | X) = \prod_{x}
\frac{\mathbb P(X = x | Y= y) \mathbb P(Y=y)}{\mathbb P(X=x)}
$$

Le problème est que si un des n-gramme est présent dans l'entrée mais qu'il n'a pas été vu à l'entrainement 
$$\mathbb P(X = x | Y= y)=0$$

et donc 

$$
\mathbb P(Y=y | X) = \prod_{x}
\frac{\mathbb P(X = x | Y= y) \mathbb P(Y=y)}{\mathbb P(X=x)} = 0
$$

Pour résoudre ce problème, nous utilisons *le lissage de Laplace* qui consiste à initier toutes les probabilités de $X$ à 1.

##### Avantages

##### Inconvénients