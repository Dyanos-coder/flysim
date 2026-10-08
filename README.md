# FlySim

Des mouches du vinaigre (*Drosophila melanogaster*) virtuelles, pilotées par la
carte complète de leur cerveau. Chaque mouche a un corps physique et son propre
cerveau : les **138 639 neurones** et **15 millions de connexions** du connectome
FlyWire. Personne ne lui dit quoi faire : elle se promène, s'arrête, mange, se
nettoie et s'enfuit parce que son cerveau, simulé neurone par neurone, en décide.
On la regarde vivre dans le navigateur, on interagit avec elle, et on voit son
cerveau s'allumer en direct.

## Lancer le projet

Il faut [Git](https://git-scm.com/) et une connexion internet pour le premier
lancement. Tout le reste (Python, dépendances, données du cerveau) s'installe
automatiquement.

```sh
git clone https://github.com/Dyanos-coder/flysim.git
cd flysim
```

Puis :

- **Windows** : double-clique sur `lancer.bat`.
- **macOS / Linux** : `./lancer.sh`

Le lanceur propose d'installer [uv](https://docs.astral.sh/uv/) (le gestionnaire
Python utilisé par le projet) s'il manque, puis ouvre la simulation dans ton
navigateur à l'adresse http://localhost:8000.

Le premier lancement prend quelques minutes : Python et les dépendances
s'installent, puis les données du cerveau (~135 Mo) se téléchargent dans
`data/`. Les lancements suivants démarrent en quelques secondes.

Si tu as déjà `uv`, une seule commande suffit :

```sh
uv run python -m flysim.server
```

Options utiles :

| Option | Effet |
|---|---|
| `--flies 3` | démarrer avec 3 mouches (on peut aussi en ajouter depuis la page, jusqu'à 6) |
| `--cpu-brain` | faire tourner les cerveaux sur le processeur même si une carte NVIDIA est là |
| `--no-brain` | physique seule, sans cerveau (pour tester) |
| `--no-browser` | ne pas ouvrir le navigateur automatiquement |
| `--port 8080` | changer le port (8000 par défaut) |

### Configuration matérielle

- **Avec une carte graphique NVIDIA** (Windows ou Linux, pilote récent compatible
  CUDA 13) : les cerveaux tournent sur la carte, détectée automatiquement. C'est
  nettement plus rapide quand il y a plusieurs mouches.
- **Sans carte NVIDIA, ou sur Mac** : tout fonctionne sur le processeur. Une mouche
  tourne à peu près en temps réel sur un portable récent ; au-delà, la simulation
  ralentit.
- Le navigateur affiche la scène en 3D (WebGL) : un navigateur récent suffit.
  Fermer les onglets gourmands aide, car il partage le processeur avec la
  simulation.

## Ce qu'on peut faire

| Outil | Ce qui se passe |
|---|---|
| ✋ Toucher | Cliquer-glisser sur la mouche pour la pousser. Toucher sa tête déclenche le toilettage. |
| 🎯 Lancer | Une balle part vers le point visé. Si elle la voit arriver, elle s'envole. |
| 🍬 Sucre | Une goutte de jus sucré. En explorant, elle finit par la trouver, la goûte, sort sa trompe et la mange. |
| 🧪 Amer | Une goutte amère : elle la goûte et n'en veut pas. |
| 🍷 Vinaigre | Une source d'odeur : on voit ses circuits olfactifs s'allumer. |
| ⚠️ Menace | Une grosse boule fonce sur elle : fuite. |
| ➕ / ➖ | Ajouter ou retirer des mouches, chacune avec son propre cerveau. |
| 🧠 Carte | La carte du cerveau de la mouche sélectionnée, en direct. |

Avec plusieurs mouches, elles partagent la nourriture et se voient entre elles :
une mouche qui s'envole près des autres peut les faire fuir à leur tour.

## Ce que décide le cerveau

Le monde traduit ce que la mouche perçoit en activité de ses neurones
sensoriels, puis lit l'activité de ses neurones descendants (ceux qui
transmettent les ordres au corps). Les liens suivants **ne sont pas programmés** :
ils émergent du câblage réel.

| Situation | Neurones sensoriels | Neurones lus | La mouche |
|---|---|---|---|
| Aucune (activité de fond) | sensoriels et visuels, activité spontanée | DNp09 (avancer), DNa02/DNa01 (tourner) | marche par épisodes, s'arrête, change de direction |
| Sucre sur la trompe ou les pattes | récepteurs du sucre | motoneurones de la trompe (CB0911, CB0871) | s'arrête, sort la trompe, mange |
| Goutte amère | récepteurs de l'amer | (motoneurones silencieux) | refuse |
| Objet qui fonce sur elle | LPLC2, détecteurs d'approche de chaque œil | fibre géante DNp01, DNa02 du côté opposé | s'envole à l'opposé |
| Contact sur la tête ou le corps | soies mécanosensorielles | neurones descendants du toilettage (DNg) | se nettoie |
| Odeur de vinaigre ou de fruit | neurones olfactifs | — | circuits olfactifs actifs |

**Ce qui reste animé par le programme.** Le cerveau modélisé s'arrête au cou : la
moelle ventrale, qui coordonne les pattes, n'est pas dans FlyWire. Les pas
rejouent donc une démarche enregistrée sur une vraie mouche, et les gestes du vol
et du toilettage sont animés. Le vol n'a pas d'aérodynamique. Enfin, le modèle ne
sait pas de quel côté vient une odeur (aucun neurone descendant ne le code) :
la mouche trouve la nourriture en explorant. Le détail est dans
[docs/autonomie.md](docs/autonomie.md).

## Comment ça marche

- **Corps et physique** : le modèle NeuroMechFly de
  [FlyGym](https://github.com/NeLy-EPFL/flygym) (issu d'un scan micro-CT d'une
  vraie mouche), simulé avec [MuJoCo](https://mujoco.org/). La marche est un
  réseau d'oscillateurs (CPG) qui rejoue des pas enregistrés, piloté par un signal
  descendant gauche/droite.
- **Cerveau** : le connectome FlyWire v783 simulé avec le modèle « leaky
  integrate-and-fire » de [Shiu et al., *Nature* 2024](https://github.com/philshiu/Drosophila_brain_model),
  réécrit pour tourner en direct : sur processeur avec numba (calcul
  événementiel, seuls les neurones actifs sont mis à jour), ou sur carte NVIDIA
  avec CuPy (tous les cerveaux ensemble, deux noyaux CUDA par pas de 0,5 ms
  rejoués en graphe CUDA).
- **Activité de fond** : comme chez une vraie mouche, les neurones sensoriels et
  les neurones de projection visuelle ont une activité spontanée (2 Hz, 0,5 Hz
  pour le goût). Elle traverse le câblage et fait fluctuer d'eux-mêmes les
  neurones de la marche.
- **Plusieurs mouches** : chacune a sa propre simulation physique (un thread par
  mouche, MuJoCo libère le verrou de Python) et son propre cerveau ; le câblage
  est partagé en mémoire. Toutes les 10 ms, le monde échange ce que les mouches
  partagent : nourriture, balles, bousculades, et ce qu'elles voient les unes des
  autres.
- **Affichage** : Three.js dans le navigateur, alimenté par un WebSocket. La
  carte du cerveau place chaque neurone à sa position FlyWire, coloré par famille,
  et le fait briller quand il émet une impulsion.

### Corrections du connectome

Tel quel, le connectome v783 s'emballe dans une activité qui s'auto-entretient
dès qu'on stimule l'odorat. Deux corrections, détaillées dans
[flysim/neurons.py](flysim/neurons.py) :

- **Principe de Dale** : un seul signe (excitateur ou inhibiteur) par neurone,
  d'après son neurotransmetteur connu dans la littérature, sinon prédit.
- **Lobe antennaire** : les neurones locaux au neurotransmetteur incertain sont
  traités comme GABAergiques (le cas majoritaire), et la sortie chimique des
  neurones locaux cholinergiques est retirée (ils agissent surtout par jonctions
  électriques, absentes du modèle).

### Validation

- [scripts/validate_brain.py](scripts/validate_brain.py) refait l'expérience du
  sucre de Shiu et al. sur FlyWire v630 et compare neurone par neurone avec leur
  modèle Brian2 : corrélation r = 0,999 (pente 1,01) au pas de 0,1 ms, et
  r = 0,996 au pas de 0,5 ms utilisé en direct (taux ~7 % plus élevés).
- [scripts/validate_gpu.py](scripts/validate_gpu.py) compare le cerveau GPU au
  cerveau CPU : r = 0,998 sur le test du sucre.
- [scripts/spontaneous.py](scripts/spontaneous.py) mesure ce que l'activité de
  fond déclenche (marche, virages, et l'absence de comportements fantômes).
- [scripts/test_behaviors.py](scripts/test_behaviors.py) vérifie de bout en bout
  manger, refuser, fuir, se nettoyer et le repas à plusieurs.

Les scripts de validation ont besoin de fichiers de référence supplémentaires :
`uv run python -m flysim.data --validation` (~90 Mo).

## Structure du code

| Chemin | Rôle |
|---|---|
| `flysim/brain.py` | Moteur du cerveau sur processeur (numba) |
| `flysim/brain_gpu.py` | Moteur du cerveau sur carte NVIDIA (CuPy) |
| `flysim/brain_link.py` | Relie un cerveau au corps : entrées sensorielles, lectures, activité de fond |
| `flysim/connectome.py` | Chargement du connectome (mis en cache en .npz) |
| `flysim/neurons.py` | Groupes de neurones nommés, corrections du connectome |
| `flysim/data.py` | Téléchargement des données au premier lancement |
| `flysim/fly.py` | Une mouche : physique, sens, cerveau → comportement, vol, toilettage |
| `flysim/world.py` | Le monde partagé : mouches, nourriture, balles, pas en parallèle |
| `flysim/locomotion.py` | Contrôleur de marche (CPG vectorisé) |
| `flysim/export.py` | Envoi de la scène MuJoCo au navigateur |
| `flysim/server.py` | Serveur : simulation, WebSocket, page web |
| `web/` | Interface Three.js |
| `scripts/` | Validation et tests |
| `docs/autonomie.md` | Ce qui vient du cerveau, ce qui reste animé, et la suite |

## Feuille de route

1. ✅ Une mouche physique qui marche, qu'on peut pousser
2. ✅ Le cerveau FlyWire décide : manger, fuir, tourner, se nettoyer
3. ✅ Carte du cerveau en direct ; nourriture, balles, vol
4. ✅ Plusieurs mouches qui interagissent
5. ✅ Mouche autonome : l'activité de fond du cerveau décide de la marche
6. ✅ Cerveaux sur carte graphique
7. Physique sur carte graphique, vraie vision (yeux à facettes → lobes optiques),
   faim et satiété, parade nuptiale

## Crédits

Ce projet s'appuie sur les travaux et les données de :

- **FlyWire** : Dorkenwald et al., « Neuronal wiring diagram of an adult brain »,
  *Nature* 2024 ; Schlegel et al., « Whole-brain annotation and multi-connectome
  cell typing of *Drosophila* », *Nature* 2024
  ([annotations](https://github.com/flyconnectome/flywire_annotations)).
- **Modèle du cerveau** : Shiu et al., « A Drosophila computational brain model
  reveals sensorimotor processing », *Nature* 2024
  ([code et données](https://github.com/philshiu/Drosophila_brain_model)).
- **NeuroMechFly / FlyGym** : Wang-Chen et al., « NeuroMechFly v2: simulating
  embodied sensorimotor control in adult *Drosophila* », *Nature Methods* 2024
  ([FlyGym](https://github.com/NeLy-EPFL/flygym)).
- **MuJoCo** (Google DeepMind), **Three.js**, **CuPy**, **numba**.

Les données téléchargées restent soumises aux licences de leurs auteurs.
