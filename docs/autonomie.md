# Vers une mouche autonome

> Objectif : que la mouche « vive sa vie » d'elle-même (marcher, s'arrêter,
> explorer, se nettoyer, chercher à manger) et qu'on vienne seulement interagir
> avec elle, sans comportement écrit à la main.

## La réponse courte

**Oui, on peut aller beaucoup plus loin**, et c'est la bonne direction : la
plupart des choses aujourd'hui scriptées peuvent être remplacées par des
mécanismes qui passent par le connectome.

**Mais « zéro script, comme une vraie mouche » n'est atteignable par personne
aujourd'hui**, y compris les laboratoires qui ont produit ces données (FlyWire,
Janelia, EPFL, DeepMind). La raison est simple : il manque encore des morceaux
de la mouche dans les données, et d'autres sont trop coûteux à simuler en temps
réel. Le plan ci-dessous remplace les scripts un par un, du plus faisable au plus
ambitieux, et dit honnêtement ce qui restera.

## Ce qui est scripté aujourd'hui

| Comportement | Aujourd'hui | Pourquoi |
|---|---|---|
| Décider de marcher | Bouton « Marcher » | Le modèle de cerveau est silencieux sans stimulation : aucun neurone ne s'active tout seul |
| Aller vers une odeur | Réflexe codé (remonte le gradient) | Dans le modèle, aucun neurone descendant ne code le côté de l'odeur |
| Vision (menace) | Je calcule moi-même l'expansion de l'objet et je stimule LPLC2 | Les yeux ne sont pas simulés ; je court-circuite les 77 000 neurones optiques |
| Gestes du toilettage | Mouvement de pattes animé | La moelle ventrale (VNC), qui coordonne les pattes, n'est pas dans FlyWire |
| Vol | Trajectoire animée, sans aérodynamique | Pas de contrôleur de vol issu du cerveau |
| Envie de manger | Absente (elle mange toujours si elle goûte du sucre) | Pas de neuromodulation (faim, satiété) dans le modèle |

Ce qui **n'est pas** scripté : la décision de manger (sucre → motoneurones de la
trompe), le refus de l'amer, la fuite (LPLC2 → fibre géante), le virage à
l'opposé d'une menace, le déclenchement du toilettage. Tout ça sort du câblage.

## Pourquoi une vraie mouche bouge toute seule

Une vraie mouche n'attend pas qu'on la stimule. Trois choses la font agir en
permanence, et toutes les trois manquent au modèle actuel :

1. **Une activité de fond.** Les neurones réels ne sont jamais au repos parfait :
   bruit synaptique, décharges spontanées, neurones qui s'activent d'eux-mêmes.
   Le modèle de Shiu et al. part d'un cerveau totalement silencieux.
2. **Un flot sensoriel continu.** Même immobile, une mouche voit (ses
   photorécepteurs répondent en permanence à la lumière), sent l'air, perçoit la
   position de ses pattes. Aujourd'hui le cerveau ne reçoit rien tant qu'on ne
   déclenche pas un stimulus.
3. **Des états internes.** Faim, satiété, fatigue, excitation : ce sont des
   neuromodulateurs (octopamine, dopamine…) qui changent la façon dont tout le
   cerveau répond. Le modèle n'a que des synapses fixes.

## Le plan, par paliers

Chaque palier remplace un script par un mécanisme qui passe par le cerveau.
Chacun commence par une **expérience de faisabilité** : si le connectome ne
produit pas le comportement, on le saura avant de construire quoi que ce soit.

### Palier 1 : activité spontanée → la mouche décide de marcher

- **Quoi** : ajouter au modèle un bruit de fond sur les neurones, et une entrée
  sensorielle tonique réaliste (décharge spontanée des neurones olfactifs,
  mécanosensoriels, gustatifs). Puis lire les **neurones descendants connus de la
  marche** et les brancher sur le contrôleur de pattes :
  - DNp09 (« P9 ») → avancer (Bidaye et al. 2020),
  - DNa02, DNa01 → tourner (Rayshubskiy et al. 2020),
  - MDN (« moonwalker ») → reculer (Bidaye et al. 2014).
- **Remplace** : le bouton « Marcher ». La mouche marcherait, s'arrêterait et
  tournerait quand son cerveau l'y pousse.
- **Ce qui reste un choix de modélisation** : l'amplitude du bruit, et la
  traduction « activité de P9 → vitesse de marche ». Ce n'est pas un script de
  comportement, c'est un paramètre de modèle, comme le poids d'une synapse, et
  il se justifie par la littérature.
- **Risque principal** : que le bruit n'active pas ces neurones de façon
  crédible (dans tous nos tests, DNp09 est resté à 0 Hz). C'est exactement ce que
  l'expérience de faisabilité vérifiera en premier.
- **Coût** : un cerveau en activité permanente ne profite plus de mon
  optimisation « seuls les neurones actifs travaillent ». Chaque cerveau
  coûterait au moins 0,7 s de calcul par seconde simulée sur ton processeur, donc
  une seule mouche en temps réel au mieux. Au-delà, il faut passer le cerveau
  sur ta carte graphique (voir « Matériel »).
- **Taille** : moyenne.

**Résultat (fait, `scripts/spontaneous.py`)** :
- Le bruit synaptique seul ne fait jamais tirer aucun neurone, et une lumière
  uniforme sur les photorécepteurs ne propage rien. Ce sont des neurones
  inhibiteurs (histamine), et ce modèle à impulsions ne sait pas calculer la
  vision à partir d'eux.
- DNp09 reçoit surtout ses entrées de LC9, LC31 et LCe04, des détecteurs
  d'objets en mouvement. Avec une activité de fond de 2 Hz sur les neurones de
  projection visuelle et les autres capteurs (0,5 Hz pour le goût), DNp09 tire
  par bouffées et DNa02/DNa01 oscillent des deux côtés. Fibre géante, trompe et
  toilettage restent sous leurs seuils : aucun comportement fantôme.
- Branché sur la marche (DNp09 lissé sur 0,8 s), la mouche marche environ 58 %
  du temps en épisodes de quelques secondes, s'arrête, repart et change de
  direction, sans aucune commande. Le bouton « Marcher » et la recherche d'odeur
  scriptée ont été retirés.
- Coût : environ 1 s de calcul par seconde simulée et par cerveau. Une mouche
  autonome tourne quasiment en temps réel ; plusieurs mouches ralentissent.

### Palier 2 : de vrais yeux → la vision passe par les 77 000 neurones optiques

- **Quoi** : FlyGym sait rendre ce que voit chaque œil à facettes
  (`get_ommatidia_readouts`). On injecte cette image dans les photorécepteurs
  du connectome (R1-R8), et on laisse les lobes optiques calculer eux-mêmes le
  mouvement, les objets qui approchent, etc. C'est l'approche de Lappalainen
  et al. (*Nature* 2024), qui ont montré qu'un modèle contraint par le
  connectome prédit l'activité réelle du système visuel.
- **Remplace** : mon calcul de looming. La fuite viendrait de ce que la mouche
  *voit* vraiment, y compris les autres mouches et ta main virtuelle.
- **Risque** : faire correspondre chaque facette aux bons photorécepteurs du
  connectome (travail de cartographie colonne par colonne), et le coût. Les
  lobes optiques en activité permanente, c'est plus de la moitié du cerveau qui
  travaille en continu.
- **Taille** : grosse. Probablement impossible en temps réel sans carte graphique.

### Palier 3 : états internes (faim, satiété)

- **Quoi** : le connectome contient des neurones connus pour signaler la faim et
  la satiété (neurones neuroendocrines, circuits de l'insuline…). On les
  stimule selon ce que la mouche a mangé : une mouche affamée devient plus
  sensible au sucre et explore plus, une mouche rassasiée se désintéresse.
- **Remplace** : « elle mange toujours dès qu'elle goûte du sucre ».
- **Limite** : la neuromodulation réelle agit sur la chimie des synapses, que ce
  modèle ne représente pas. On ne peut l'approcher qu'en stimulant ces neurones.
- **Taille** : moyenne.

### Palier 4 : l'odeur par le cerveau

- **Quoi** : supprimer le réflexe codé et laisser le cerveau décider. Honnêtement,
  avec ce modèle, **la mouche ne trouverait probablement plus la nourriture** :
  il ne code pas le côté de l'odeur. Une vraie mouche s'aide du vent et des
  variations dans le temps, que le modèle ne capture pas.
- **Option** : l'activité spontanée du palier 1 donne déjà une exploration
  aléatoire. La mouche finirait par tomber sur la nourriture par hasard, puis le
  goût ferait le reste via le cerveau. C'est moins spectaculaire, mais sans triche.
- **Taille** : petite (c'est surtout retirer du code).

### Palier 5 : les pattes par le connectome (frontière de la recherche)

- **Quoi** : ajouter la moelle ventrale. Des connectomes récents la contiennent
  (MANC, MaleCNS chez le mâle, BANC pour le système nerveux complet d'une
  femelle), à vérifier au moment de s'y lancer. Des travaux récents commencent à
  y identifier les circuits qui rythment la marche. On pourrait alors remplacer
  les pas enregistrés et le toilettage animé par des motoneurones qui pilotent
  les articulations.
- **Pourquoi c'est la frontière** : il faut modéliser les muscles, et le passage
  « motoneurones → forces dans les articulations » n'est pas résolu, même dans
  les labos. Aucune démonstration publique ne fait marcher une mouche simulée
  uniquement à partir de son connectome complet.
- **Taille** : projet de recherche. Plusieurs mois, sans garantie de résultat.

### Le vol

Même DeepMind (flybody) fait voler sa mouche avec un réseau entraîné par
apprentissage par renforcement, pas avec son connectome. Un vol « sans script »
n'existe nulle part aujourd'hui. Options : garder le vol animé, ou **retirer le
vol** et ne garder que le saut de fuite, qui vient bien du cerveau (fibre géante).

## Matériel

| Ce qu'on fait | Processeur (actuel) | Carte graphique RTX 3050 |
|---|---|---|
| Cerveaux calmes, stimulés à la demande (aujourd'hui) | OK jusqu'à ~3 mouches | inutile |
| Palier 1, activité permanente | ~1 mouche en temps réel au mieux | plusieurs mouches |
| Palier 2, vraie vision | trop lent | probablement 1 mouche |

Passer le cerveau sur la carte graphique demande de réécrire le moteur neuronal
(PyTorch ou CuPy) et un téléchargement de 0,5 à 2,5 Go.

## Ce que je propose

1. **Expérience de faisabilité du palier 1** : ajouter bruit et entrées toniques,
   et mesurer si DNp09, DNa01/DNa02 et MDN s'activent spontanément, de façon
   crédible (ni tout le temps, ni jamais), et à quel coût de calcul.
2. Si oui : brancher ces neurones sur la marche, retirer le bouton « Marcher » et
   le réflexe olfactif. La mouche vit sa vie ; on interagit.
3. Si le coût est trop élevé : porter le moteur neuronal sur la carte graphique.
4. Ensuite, selon le résultat : états internes (palier 3), puis vraie vision
   (palier 2).

## Ce que ça change pour la démo

Avec les paliers 1 et 3, le message devient : *« Personne ne lui dit quoi faire.
Elle se promène, s'arrête, se nettoie, cherche à manger, et fuit quand on la
menace, parce que son cerveau, reconstruit neurone par neurone, en décide. »*
Ce sera vrai, à l'exception clairement identifiée des gestes eux-mêmes (pattes,
ailes), qui restent animés tant que la moelle ventrale n'est pas modélisée.
