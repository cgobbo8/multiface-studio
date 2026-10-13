# MultiFace Studio

**Un nouveau visage pour chaque personne d’une photo ou d’une vidéo, en une seule passe.**

FaceFusion remplace un visage à la fois. Avec 20 personnes sur une image, il faut donc lancer 20 rendus de suite.
MultiFace Studio détecte toutes les personnes, vous laisse déposer un visage source en face de chacune, puis génère tout d’un coup.
L’interface est inspirée d’Apple (verre dépoli, thème clair/sombre automatique).

Le moteur est [FaceFusion 3.5.4](https://github.com/facefusion/facefusion), avec un petit patch (`app/patches/`).

> Projet communautaire **non officiel**, sans lien avec l’équipe FaceFusion.

## Fonctionnalités

- **Détection de toutes les personnes** : sur une photo, chaque visage ; sur une vidéo, le studio parcourt toute la vidéo et regroupe les apparitions d’une même personne, même de profil (empreintes faciales, plusieurs angles par personne).
- **Un visage source par personne** : bibliothèque de visages, glisser-déposer ou « sélectionner puis toucher », répartition automatique dans l’ordre. Les personnes sans source restent intactes.
- **Attribution un-pour-un** : deux personnes qui se ressemblent ne reçoivent jamais la même source sur une même image.
- **Aperçu et comparaison avant/après** sur n’importe quelle image de la vidéo.
- **Extrait vidéo** (début/fin), progression du rendu, annulation.
- **Téléchargement des modèles visible** : nom, pourcentage et Mo reçus.
- Réglages : modèle de remplacement, résolution, force du visage source, amélioration (GFPGAN, CodeFormer…), respect des occlusions, tolérance de reconnaissance, détecteur, qualité d’export, accélération matérielle.

## Installation (Pinokio)

1. Dans Pinokio, **Download from URL** → collez `https://github.com/cgobbo8/multiface-studio`.
2. Cliquez sur **Installer**. Le script clone FaceFusion 3.5.4, applique le patch et crée un environnement conda (Python 3.12, ffmpeg, onnxruntime adapté à votre GPU).
3. Cliquez sur **Démarrer**, puis **Ouvrir le studio**.

Si le lanceur officiel FaceFusion est déjà installé dans Pinokio (`api/facefusion-pinokio.git`), ses modèles sont réutilisés via un lien : rien n’est retéléchargé.
Sinon, les modèles de base (~1 Go) sont téléchargés au premier démarrage, avec la progression affichée dans le studio.

- **Mettre à jour** : récupère la dernière version du lanceur et réapplique le patch.
- **Réinitialiser** : supprime le moteur et l’environnement. Les modèles partagés avec le lanceur officiel ne sont pas touchés.

## Utilisation

1. **Ouvrir** une photo ou une vidéo (ou la glisser dans la fenêtre). Pour une vidéo, l’analyse de toutes les personnes démarre automatiquement.
2. Ajoutez les photos des nouveaux visages dans **Visages sources**.
3. Associez un visage source à chaque personne voulue :
   - glissez le visage sur le « + » de la personne ;
   - ou touchez le visage, puis la personne (dans la liste ou directement sur l’image).
4. Vérifiez avec **Aperçu** ou **Comparer**.
5. **Générer**. Le résultat est enregistré dans `app/workspace/outputs/`.

Une personne a été manquée dans une vidéo ? Placez-vous sur une image où elle apparaît et touchez **Cette image**.
Sur une grande photo de groupe avec de petits visages, baissez le **Seuil de détection** ou choisissez le détecteur **Tous**.

## API

Le studio est piloté par une API HTTP locale (le port est affiché au démarrage, ici `7870`). Tout ce que fait l’interface est accessible par script.

| Méthode | Route | Rôle |
| --- | --- | --- |
| `POST` | `/api/target` | Importer la cible (champ `file`) ; détecte les visages de la première image |
| `POST` | `/api/scan` | `{ "mode": "video", "samples": 60 }` analyse toute la vidéo (tâche) ; `{ "mode": "frame", "frame_number": 120 }` ajoute les personnes d’une image |
| `POST` | `/api/library` | Ajouter des visages sources (champ `files`, plusieurs fichiers) |
| `POST` | `/api/people/{id}/source` | `{ "source_id": "…" }` associe une source à une personne (`null` pour la retirer) |
| `POST` | `/api/people/assign-in-order` | Répartit les sources dans l’ordre |
| `POST` | `/api/settings` | Modifie les réglages (voir `settings` dans `/api/state`) |
| `POST` | `/api/trim` | `{ "start": 0, "end": 300 }` extrait vidéo, en numéros d’image |
| `GET` | `/api/preview?frame_number=0` | Aperçu JPEG d’une image |
| `POST` | `/api/render` | Lance le rendu (tâche) |
| `GET` | `/api/task` | Tâche en cours : `status`, `progress`, `stage`, `download`, `result` |
| `POST` | `/api/task/cancel` | Annule la tâche en cours |
| `GET` | `/api/output/{name}` | Télécharge un résultat |
| `GET` | `/api/state` | État complet : cible, personnes, bibliothèque, réglages, tâche |

### curl

```bash
API=http://127.0.0.1:7870
curl -s -F "file=@groupe.jpg" $API/api/target
curl -s -F "files=@alice.jpg" -F "files=@bob.jpg" $API/api/library
curl -s -X POST $API/api/people/assign-in-order
curl -s -X POST $API/api/render
curl -s $API/api/task
```

### Python

```python
import time
import requests

api = 'http://127.0.0.1:7870'

with open('groupe.jpg', 'rb') as target:
	session = requests.post(api + '/api/target', files = { 'file': target }).json()

with open('alice.jpg', 'rb') as alice, open('bob.jpg', 'rb') as bob:
	library = requests.post(api + '/api/library', files = [ ('files', alice), ('files', bob) ]).json()

for person, source_id in zip(session['people'], library['added']):
	requests.post(api + '/api/people/' + str(person['id']) + '/source', json = { 'source_id': source_id })

requests.post(api + '/api/render')

while (task := requests.get(api + '/api/task').json())['status'] == 'running':
	time.sleep(1)

print(task['status'], task['result'])
```

### JavaScript

```javascript
const api = 'http://127.0.0.1:7870';

async function render(targetFile, sourceFiles)
{
	const targetForm = new FormData();
	targetForm.append('file', targetFile);
	const session = await (await fetch(api + '/api/target', { method: 'POST', body: targetForm })).json();

	const libraryForm = new FormData();
	sourceFiles.forEach(file => libraryForm.append('files', file));
	await fetch(api + '/api/library', { method: 'POST', body: libraryForm });
	await fetch(api + '/api/people/assign-in-order', { method: 'POST' });
	await fetch(api + '/api/render', { method: 'POST' });

	let task;
	do
	{
		await new Promise(resolve => setTimeout(resolve, 1000));
		task = await (await fetch(api + '/api/task')).json();
	}
	while (task.status === 'running');

	return task.result;
}
```

## Ce que change le patch du moteur

- `facefusion/face_mapper.py` (nouveau) : associe chaque visage détecté à la personne la plus proche, avec une attribution un-pour-un.
- `face_selector.py` : mode de sélection interne `mapping`.
- `processors/modules/face_swapper/core.py` : en mode `mapping`, chaque visage reçoit son propre visage source.
- `video_manager.py` : lecture vidéo protégée par un verrou ; corrige un plantage aléatoire de libavcodec (`pthread_frame.c`, `async_lock`) quand plusieurs threads lisent la vidéo.
- `types.py`, `choices.py` : types du mode `mapping` (non exposé dans l’interface Gradio d’origine).

## Licence et usage responsable

Voir [LICENSE.md](LICENSE.md).

- FaceFusion est distribué sous licence **OpenRAIL-AS** (© Henry Ruhs). Le patch est une modification de FaceFusion et suit cette licence, y compris ses restrictions d’usage.
- Les modèles ne sont pas inclus : FaceFusion les télécharge depuis ses propres serveurs, chacun avec sa licence (plusieurs sont **non commerciaux**, par exemple InSwapper, SimSwap, BlendSwap).
- L’analyse de contenu de FaceFusion (NSFW) reste active.
- N’utilisez que des visages de personnes qui ont donné leur accord, et ne présentez jamais un résultat comme authentique.
