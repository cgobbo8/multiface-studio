# MultiFace Studio

**A new face for every person in a photo or a video, in a single pass.**

FaceFusion swaps one face at a time: with 20 people in a shot, you have to run 20 renders in a row.
MultiFace Studio detects everyone, lets you drop a source face next to each person, then renders everything at once.
The interface comes in English and French (**Settings → Appearance → Language**) and takes its cues from Apple (frosted glass, automatic light/dark theme).

This repository is a **fork of [FaceFusion](https://github.com/facefusion/facefusion)**, based on its 3.5.4 release: the multi-face engine changes are commits on top of it ([see the difference](https://github.com/facefusion/facefusion/compare/3.5.4...cgobbo8:multiface-studio:main)), the studio lives in `studio/` and the Pinokio launcher scripts at the root.

> **Unofficial** community project, not affiliated with the FaceFusion team.

## Features

- **Every person detected**: each face in a photo; in a video, the studio goes through the whole clip and groups the appearances of the same person, even in profile (face embeddings, several angles per person).
- **One source face per person**: face library, drag and drop or "select, then tap", automatic assignment in order. People without a source stay untouched.
- **Images pasted on heads (memes)**: any image (a transparent PNG works best: emoji, cut-out head, logo…) follows a person's head, with its position and tilt. The image also turns in **3D** when the head turns or nods (pose estimated from the face detector's 5 points, the same perspective as the preview's CSS 3D, adjustable strength). In videos, the image motion is lightly smoothed, never across a cut. Size, offset, rotation and mirroring are set per person with a live preview, and the shape can be whole, round or **cut out** (the background and shoulders are removed automatically by the MODNet model, 25 MB, downloaded on first use). A plain photo of someone is cropped around their head automatically. Face swaps and images can be mixed in the same render.
- **Lying or upside-down faces**: the detector only recognises upright faces, so the studio also searches the frame turned 90° both ways. A person lying down gets detected and swapped, and pasted images follow their head. It can be turned off in **Settings → Recognition** (detection is a bit slower with it).
- **One-to-one assignment**: two lookalikes never get the same source in the same frame.
- **Persistence in videos**: when detection misses a face for a few frames (blur, a hand in front), the swap or the pasted image carries on (up to 0.25 s), only if the head stayed almost in place. A face that is detected but not recognised (profile) is attached to the right person. Nothing is ever filled in across a cut or onto another person.
- **Preview and before/after comparison** on any frame of the video.
- **Live camera**: the same per-person faces and pasted images on your webcam, in real time, with a clean window to capture in OBS and send to Meet, Zoom or Slack through OBS's virtual camera.
- **Video clip**: under the video, a filmstrip with two handles to drag (as in Photos); the **I** and **O** keys set the start and the end at the current frame. Only the clip gets analysed and rendered.
- Render progress and cancellation.
- **Visible model downloads**: name, percentage and MB received.
- Settings: swap model, resolution, source face strength, enhancement (GFPGAN, CodeFormer…), occlusion handling, recognition tolerance, detector, export quality, hardware acceleration.

## Installation (Pinokio)

1. In Pinokio, **Download from URL** → paste `https://github.com/cgobbo8/multiface-studio`.
2. Click **Install**. The script creates a conda environment (Python 3.12, ffmpeg, onnxruntime matching your GPU) and installs FaceFusion's dependencies.
3. Click **Start**, then **Open Web UI**.

If the official FaceFusion launcher is already installed in Pinokio (`api/facefusion-pinokio.git`), its models are reused through a link: nothing is downloaded again.
Otherwise the base models (~1 GB) are downloaded on the first start, with their progress shown in the studio.

- **Update**: pulls the latest version (engine and studio together).
- **Reset**: removes the environment. Your files in `workspace/` and the models shared with the official launcher are left alone.

## Usage

1. **Open** a photo or a video (or drag it into the window). For a video, the analysis of every person starts automatically.
   To switch to another file: **Change video** / **Change photo** (⌘O or Ctrl+O), or drag the new file onto the frame. If sources are already assigned, you are asked to confirm before they are removed; the source library is kept.
2. In **Sources**, add photos of the new faces (**Faces** tab) and/or images to paste (**Images** tab).
3. Assign a source to each person you want to change:
   - drag it onto the person's "+";
   - or tap it, then tap the person (in the list or right on the frame).
   - For an image, tap the **Adjust** button (sliders icon) on the person: size, position, rotation, 3D effect, mirror and shape (whole, round, cut out). The **Preview** tab opens and the image updates live.
4. Check the result with **Preview** or **Compare**.
5. **Generate**. The result is saved in `workspace/outputs/`.

To keep only part of a video, drag the handles of the filmstrip under the video, or go to a frame and press **I** (start) or **O** (end).
### Live camera

Click **Live** (or **Use my camera** on the start screen) and allow the camera. The people in front of the camera are detected on their own; give them faces or images like for a photo, the stream updates on the next frame.

- **Fluid** uses the fast InSwapper 128 model, **Sharp** the model picked in the settings (nicer, slower). On an M4 Pro, expect about 7–13 frames per second with one swapped face and 30+ with pasted images only, depending on the machine's load.
- **Mirror** flips your own view like a selfie camera; what the others see is never mirrored.
- **Window for OBS** opens the result alone in a window. In OBS, add a *Window Capture* of it, then click **Start Virtual Camera** and pick *OBS Virtual Camera* in Meet, Zoom or Slack.
- The camera runs in your browser: if Pinokio's window does not offer it, open the studio's address in Chrome or Safari. The live mode is disabled when the studio runs with `--remote` (a shared server).

Only wear the face of someone who agreed to it, and make sure the people you talk to know they are looking at a filter.

Someone was missed in a video? Go to a frame where they appear and tap **This frame**.
On a large group photo with small faces, lower the **Detection threshold** or pick the **All** detector.

## API

The studio is driven by a local HTTP API (the port is printed at startup, `7870` here). Everything the interface does can be scripted.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/target` | Imports the target (`file` field) and detects the faces of its first frame |
| `POST` | `/api/scan` | `{ "mode": "video", "samples": 60 }` analyses the whole video (task); `{ "mode": "frame", "frame_number": 120 }` adds the people of one frame |
| `POST` | `/api/library?kind=face` | Adds source faces (`files` field, several files) |
| `POST` | `/api/library?kind=sticker` | Adds images to paste on heads |
| `POST` | `/api/people/{id}/source` | `{ "source_id": "…" }` assigns a source to a person (`null` removes it) |
| `POST` | `/api/people/{id}/sticker` | Sets the pasted image: `scale`, `offset_x`, `offset_y` (in face sizes), `rotation` (°), `follow_rotation`, `depth` (3D effect, 0–1), `mirror`, `round`, `cutout`, or `{ "reset": true }` |
| `POST` | `/api/people/assign-in-order?kind=face` | Assigns the sources (`face` or `sticker`) in order |
| `POST` | `/api/settings` | Changes the settings (see `settings` in `/api/state`) |
| `POST` | `/api/trim` | `{ "start": 0, "end": 300 }` video clip, in frame numbers (end excluded, `null` = edge of the video) |
| `GET` | `/api/preview?frame_number=0` | JPEG preview of a frame |
| `POST` | `/api/render` | Starts the render (task) |
| `GET` | `/api/task` | Current task: `status`, `progress`, `stage`, `message`, `download`, `result` |
| `POST` | `/api/task/cancel` | Cancels the current task |
| `GET` | `/api/output/{name}` | Downloads a result |
| `GET` | `/api/state` | Full state: target, people, library, settings, task |
| `POST` | `/api/live/start` | `{ "width": 640, "height": 360 }` switches the target to the live camera (people and sources are kept when it already was) |
| `WS` | `/api/live/stream` | Send JPEG camera frames as binary messages; each one comes back as a JSON text message (`faces` with `box` and `person_id`, `flagged`, `ms`) followed by the processed JPEG |

Texts sent by the server (a task's `stage` and `message`, a refused request's `error`) are translation keys, for example `stage.swap`, or `message.frames` with its values in `message_params`. The English and French texts live in `studio/static/i18n.js`.

### curl

```bash
API=http://127.0.0.1:7870
curl -s -F "file=@group.jpg" $API/api/target
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

with open('group.jpg', 'rb') as target:
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

## What this fork changes in FaceFusion

Everything below sits in `facefusion/`, as commits on top of the 3.5.4 release:


- `facefusion/face_mapper.py` (new): matches each detected face with the closest person, one-to-one.
- `face_selector.py`: internal `mapping` selector mode.
- `processors/modules/face_swapper/core.py`: in `mapping` mode, each face gets its own source face.
- `processors/modules/face_sticker/` (new): a processor that pastes an image on the head of the mapped people, following their position, size, tilt and 3D rotation (pose estimated from the detector's 5 points, perspective projection, smoothing within each shot that stays correct for upside-down heads).
- `face_mapper.py`: in videos, a first pass tracks every person through the whole video and carefully fills short gaps; the swapper, the enhancer and the pasted images rely on it.
- `workflows/image_to_video.py`: processors can prepare the whole video before rendering (`prepare_frames`) and receive the number of the frame being processed.
- `video_manager.py`: video reads are guarded by a lock, which fixes a random libavcodec crash (`pthread_frame.c`, `async_lock`) when several threads read the video.
- `types.py`, `choices.py`: types of the `mapping` mode (not exposed in the original Gradio interface).

## License and responsible use

See [LICENSE.md](LICENSE.md).

- FaceFusion is distributed under the **OpenRAIL-AS** license (© Henry Ruhs). This fork and its changes (© 2026 Corentin Gobbo) follow the same license, including its use-based restrictions.
- Models are not included: FaceFusion downloads them from its own servers, each under its own license (several are **non-commercial**, for example InSwapper, SimSwap and BlendSwap).
- FaceFusion's content analyser (NSFW) stays active.
- Only use the faces of people who gave their consent, and never present a result as authentic.
