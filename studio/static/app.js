'use strict';

const $ = (selector) => document.querySelector(selector);

const el =
{
	targetCaption: $('#target-caption'),
	openTarget: $('#open-target'),
	openTargetLabel: $('#open-target-label'),
	openSettings: $('#open-settings'),
	render: $('#render'),
	renderLabel: $('#render-label'),
	modeControl: $('#mode-control'),
	viewerHint: $('#viewer-hint'),
	canvas: $('#canvas'),
	dropzone: $('#dropzone'),
	pickTarget: $('#pick-target'),
	frame: $('#frame'),
	frameOriginal: $('#frame-original'),
	frameResult: $('#frame-result'),
	faceLayer: $('#face-layer'),
	stickerLayer: $('#sticker-layer'),
	compareHandle: $('#compare-handle'),
	frameSpinner: $('#frame-spinner'),
	canvasDrop: $('#canvas-drop'),
	canvasDropText: $('#canvas-drop-text'),
	timeline: $('#timeline'),
	scrubber: $('#scrubber'),
	scrubberTrim: $('#scrubber-trim'),
	stepBack: $('#step-back'),
	stepForward: $('#step-forward'),
	timecode: $('#timecode'),
	trimStart: $('#trim-start'),
	trimEnd: $('#trim-end'),
	trimReset: $('#trim-reset'),
	library: $('#library'),
	librarySt: $('#library-strip'),
	libraryKind: $('#library-kind'),
	libraryHint: $('#library-hint'),
	addSources: $('#add-sources'),
	peopleCount: $('#people-count'),
	peopleSubtitle: $('#people-subtitle'),
	scanFrame: $('#scan-frame'),
	scanVideo: $('#scan-video'),
	scanProgress: $('#scan-progress'),
	scanBar: $('#scan-bar'),
	scanText: $('#scan-text'),
	scanCancel: $('#scan-cancel'),
	peopleList: $('#people-list'),
	peopleEmpty: $('#people-empty'),
	peopleFooter: $('#people-footer'),
	assignOrder: $('#assign-order'),
	clearSources: $('#clear-sources'),
	settings: $('#settings'),
	settingsBody: $('#settings-body'),
	settingsBackdrop: $('#settings-backdrop'),
	closeSettings: $('#close-settings'),
	hud: $('#hud'),
	hudRing: $('#hud-ring'),
	hudPercent: $('#hud-percent'),
	hudStage: $('#hud-stage'),
	hudMessage: $('#hud-message'),
	hudCancel: $('#hud-cancel'),
	result: $('#result'),
	resultName: $('#result-name'),
	resultMedia: $('#result-media'),
	closeResult: $('#close-result'),
	revealResult: $('#reveal-result'),
	revealLabel: $('#reveal-label'),
	downloadResult: $('#download-result'),
	confirm: $('#confirm'),
	confirmTitle: $('#confirm-title'),
	confirmMessage: $('#confirm-message'),
	confirmAccept: $('#confirm-accept'),
	confirmCancel: $('#confirm-cancel'),
	dropOverlay: $('#drop-overlay'),
	dropOverlayText: $('#drop-overlay-text'),
	toasts: $('#toasts'),
	offline: $('#offline'),
	downloadCard: $('#download-card'),
	downloadPercent: $('#download-percent'),
	downloadName: $('#download-name'),
	downloadBar: $('#download-bar'),
	downloadSize: $('#download-size'),
	hudRingBox: $('#hud-ring-box'),
	offlineRetry: $('#offline-retry'),
	targetInput: $('#target-input'),
	sourcesInput: $('#sources-input'),
	slotInput: $('#slot-input')
};

const state =
{
	session: null,
	options: null,
	frameNumber: 0,
	mode: 'original',
	faces: [],
	armedId: null,
	hoverPersonId: null,
	slotPersonId: null,
	comparePosition: 0.5,
	previewSeq: 0,
	facesSeq: 0,
	previewUrl: null,
	previewDirty: true,
	frameTimer: null,
	settingsTimers: {},
	pollTimer: null,
	lastHandledTaskId: null,
	renderedPersonIds: new Set(),
	importing: false,
	libraryKind: 'face',
	adjustPersonId: null,
	stickerTimers: {},
	offline: false,
	reconnectTimer: null,
	frameErrorTimer: null
};

const ICONS =
{
	plus: '<svg viewBox="0 0 24 24"><path d="M12 6v12M6 12h12"/></svg>',
	close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
	chevron: '<svg class="person-arrow" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>',
	sliders: '<svg viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/></svg>'
};

const LIBRARY_KINDS =
{
	face:
	{
		hint: 'library.face_hint',
		empty: 'library.face_empty',
		added: 'library.face_added',
		rejected: 'library.face_rejected'
	},
	sticker:
	{
		hint: 'library.sticker_hint',
		empty: 'library.sticker_empty',
		added: 'library.sticker_added',
		rejected: 'library.sticker_rejected'
	}
};

const LIBRARY_MIME = 'application/x-facefusion-source';

/* ------------------------------------------------------------ api */

function offlineError()
{
	const error = new Error(t('offline.message'));

	error.offline = true;
	return error;
}

async function request(method, url, body)
{
	const options = { method, headers: {} };

	if (body !== undefined)
	{
		options.headers['Content-Type'] = 'application/json';
		options.body = JSON.stringify(body);
	}
	let response;

	try
	{
		response = await fetch(url, options);
	}
	catch (error)
	{
		setOffline(true);
		throw offlineError();
	}
	const data = await response.json().catch(() => ({}));

	if (!response.ok)
	{
		throw new Error(data.error ? translateServer(data.error) : t('error.status', { status: response.status }));
	}
	return data;
}

function upload(url, files, field, onProgress)
{
	return new Promise((resolve, reject) =>
	{
		const form = new FormData();
		const xhr = new XMLHttpRequest();

		Array.from(files).forEach(file => form.append(field, file, file.name));
		xhr.open('POST', url);
		xhr.responseType = 'json';
		xhr.upload.onprogress = event =>
		{
			if (event.lengthComputable && onProgress)
			{
				onProgress(event.loaded / event.total);
			}
		};
		xhr.upload.onload = () => onProgress && onProgress(1);
		xhr.onload = () =>
		{
			if (xhr.status >= 200 && xhr.status < 300)
			{
				resolve(xhr.response);
			}
			else
			{
				reject(new Error(xhr.response && xhr.response.error ? translateServer(xhr.response.error) : t('error.status', { status: xhr.status })));
			}
		};
		xhr.onerror = () =>
		{
			setOffline(true);
			reject(offlineError());
		};
		xhr.send(form);
	});
}

/* ------------------------------------------------------------ helpers */

function target()
{
	return state.session && state.session.target;
}

function people()
{
	return (state.session && state.session.people) || [];
}

function library()
{
	return (state.session && state.session.library) || [];
}

function isVideo()
{
	return Boolean(target() && target().type === 'video');
}

function isBusy()
{
	return Boolean(state.session && state.session.task && state.session.task.status === 'running');
}

function assignedCount()
{
	return people().filter(person => person.source_id).length;
}

function personIndex(personId)
{
	return people().findIndex(person => person.id === personId);
}

function escapeHtml(value)
{
	return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '\'': '&#39;' })[character]);
}

function formatTimecode(frameNumber)
{
	const fps = (target() && target().fps) || 25;
	const seconds = frameNumber / fps;
	const minutes = Math.floor(seconds / 60);
	const rest = seconds - minutes * 60;
	return t('timeline.timecode', { time: String(minutes).padStart(2, '0') + ':' + rest.toFixed(2).padStart(5, '0'), frame: frameNumber + 1, total: target().frame_total });
}

function shortcutLabel(key)
{
	return (/Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl+') + key;
}

function toast(message, kind = 'success', duration = 3400)
{
	const node = document.createElement('div');

	node.className = 'toast is-' + kind;
	node.innerHTML = '<span class="toast-dot"></span><span></span>';
	node.lastChild.textContent = message;
	el.toasts.appendChild(node);
	setTimeout(() =>
	{
		node.classList.add('is-leaving');
		setTimeout(() => node.remove(), 320);
	}, duration);
}

function handleError(error)
{
	// the offline banner already explains a stopped server
	if (state.offline || (error && error.offline))
	{
		return;
	}
	toast(error.message || String(error), 'error', 5200);
}

/* ------------------------------------------------------------ session */

function setSession(session)
{
	state.session = session;
	render();
}

function render()
{
	renderToolbar();
	renderViewerChrome();
	renderPeople();
	renderLibrary();
	renderFaces();
	renderTask();
}

function renderToolbar()
{
	const count = assignedCount();

	el.targetCaption.textContent = target() ? target().name + ' · ' + target().width + '×' + target().height : t('app.caption');
	// once a target is loaded the open button says what it really does: replace it
	el.openTarget.classList.toggle('is-replace', Boolean(target()));
	el.openTargetLabel.textContent = target() ? t(isVideo() ? 'toolbar.replace_video' : 'toolbar.replace_image') : t('toolbar.open');
	el.openTarget.title = t(target() ? 'toolbar.replace_title' : 'toolbar.open_title') + ' (' + shortcutLabel('O') + ')';
	el.openTarget.disabled = isBusy();
	el.renderLabel.textContent = count ? t('toolbar.render_count', { count }) : t('toolbar.render');
	el.render.disabled = !target() || !count || isBusy();
}

/* ------------------------------------------------------------ viewer */

function renderViewerChrome()
{
	const hasTarget = Boolean(target());

	el.dropzone.hidden = hasTarget;
	el.frame.hidden = !hasTarget;
	el.timeline.hidden = !isVideo();
	el.scanVideo.hidden = !isVideo();
	el.scanFrame.disabled = !hasTarget || isBusy();
	el.scanVideo.disabled = isBusy();

	if (isVideo())
	{
		el.scrubber.max = Math.max(0, target().frame_total - 1);
		el.scrubber.value = state.frameNumber;
		el.timecode.textContent = formatTimecode(state.frameNumber);
		renderTrim();
	}
	renderMode();
	layoutFrame();
}

function renderMode()
{
	const buttons = Array.from(el.modeControl.querySelectorAll('button'));
	const activeButton = buttons.find(button => button.dataset.mode === state.mode);
	const thumb = el.modeControl.querySelector('.segmented-thumb');

	buttons.forEach(button =>
	{
		button.classList.toggle('is-active', button === activeButton);
		button.disabled = !target();
	});
	placeSegmentThumb(el.modeControl, activeButton);

	el.frameResult.hidden = state.mode === 'original';
	el.faceLayer.hidden = state.mode !== 'original';
	renderStickerLayer();
	el.compareHandle.hidden = state.mode !== 'compare';
	updateCompare();

	if (!target())
	{
		el.viewerHint.textContent = '';
	}
	else if (state.armedId)
	{
		el.viewerHint.textContent = t('hint.armed');
	}
	else if (state.mode === 'original')
	{
		el.viewerHint.textContent = people().length ? t('hint.original') : '';
	}
	else if (!assignedCount())
	{
		el.viewerHint.textContent = t('hint.no_source');
	}
	else
	{
		el.viewerHint.textContent = state.mode === 'compare' ? t('hint.compare') : t('hint.result');
	}
}

function placeSegmentThumb(control, activeButton)
{
	const thumb = control.querySelector('.segmented-thumb');

	thumb.style.width = activeButton.offsetWidth + 'px';
	thumb.style.transform = 'translateX(' + (activeButton.offsetLeft - 2) + 'px)';
}

function setMode(mode)
{
	state.mode = mode;
	renderMode();

	// a preview computed without stickers (live adjusting) cannot stand in for the baked one, and the other way round
	if (mode !== 'original' && (state.previewDirty || state.previewLive !== isLiveSticker()))
	{
		fetchPreview();
	}
}

function updateCompare()
{
	const percent = (state.comparePosition * 100).toFixed(2);

	el.frameResult.style.clipPath = state.mode === 'compare' ? 'inset(0 0 0 ' + percent + '%)' : '';
	el.compareHandle.style.left = percent + '%';
}

function layoutFrame()
{
	const stacked = window.matchMedia('(max-width: 1080px)').matches;

	if (!target())
	{
		el.canvas.style.height = '';
		return;
	}
	const rect = el.canvas.getBoundingClientRect();
	const padding = 18;
	const ratio = target().width / target().height;
	const availableWidth = Math.max(80, rect.width - padding * 2);
	// stacked layouts have no fixed viewer height, so the frame sizes the canvas instead of the reverse
	const availableHeight = stacked ? Math.max(220, window.innerHeight * 0.7) : Math.max(80, rect.height - padding * 2);
	let width = availableWidth;
	let height = width / ratio;

	if (height > availableHeight)
	{
		height = availableHeight;
		width = height * ratio;
	}
	el.canvas.style.height = stacked ? Math.round(height + padding * 2) + 'px' : '';
	el.frame.style.width = Math.round(width) + 'px';
	el.frame.style.height = Math.round(height) + 'px';
}

function frameUrl(frameNumber)
{
	return '/api/frame?frame_number=' + frameNumber + '&max_size=1920&v=' + target().id;
}

function showFrame()
{
	if (!target())
	{
		return;
	}
	el.frameOriginal.src = frameUrl(state.frameNumber);
	state.faces = [];
	state.previewDirty = true;
	renderFaces();

	if (isVideo())
	{
		el.scrubber.value = state.frameNumber;
		el.timecode.textContent = formatTimecode(state.frameNumber);
	}
	clearTimeout(state.frameTimer);
	state.frameTimer = setTimeout(() =>
	{
		fetchFaces();

		if (state.mode !== 'original')
		{
			fetchPreview();
		}
	}, 240);
}

function goToFrame(frameNumber)
{
	if (!isVideo())
	{
		return;
	}
	state.frameNumber = Math.max(0, Math.min(target().frame_total - 1, frameNumber));
	showFrame();
}

async function fetchFaces()
{
	if (!target() || isBusy())
	{
		return;
	}
	const sequence = ++state.facesSeq;

	try
	{
		const data = await request('GET', '/api/faces?frame_number=' + state.frameNumber);

		if (sequence === state.facesSeq)
		{
			state.faces = data.faces;
			renderFaces();
		}
	}
	catch (error)
	{
		/* the overlay is optional, a busy engine simply skips it */
	}
}

async function fetchPreview()
{
	if (!target() || isBusy())
	{
		return;
	}
	const sequence = ++state.previewSeq;

	el.frameSpinner.hidden = false;
	try
	{
		const live = isLiveSticker();
		const response = await withDownloadWatch(() => fetch('/api/preview?frame_number=' + state.frameNumber + '&max_size=1920' + (live ? '&stickers=0' : ''))).catch(() =>
		{
			setOffline(true);
			throw offlineError();
		});

		if (!response.ok)
		{
			const data = await response.json().catch(() => ({}));
			throw new Error(data.error ? translateServer(data.error) : t('error.preview'));
		}
		const blob = await response.blob();

		if (sequence !== state.previewSeq)
		{
			return;
		}
		if (state.previewUrl)
		{
			URL.revokeObjectURL(state.previewUrl);
		}
		state.previewUrl = URL.createObjectURL(blob);
		state.previewDirty = false;
		state.previewLive = live;
		el.frameResult.src = state.previewUrl;

		if (response.headers.get('X-Content-Flagged') === '1')
		{
			toast(t('toast.flagged'), 'info', 5000);
		}
	}
	catch (error)
	{
		if (sequence === state.previewSeq)
		{
			handleError(error);
		}
	}
	finally
	{
		if (sequence === state.previewSeq)
		{
			el.frameSpinner.hidden = true;
		}
	}
}

function invalidatePreview()
{
	state.previewDirty = true;

	if (state.mode !== 'original')
	{
		clearTimeout(state.previewTimer);
		state.previewTimer = setTimeout(fetchPreview, 200);
	}
}

function renderFaces()
{
	const fragment = document.createDocumentFragment();

	state.faces.forEach(face =>
	{
		const index = personIndex(face.person_id);
		const person = people()[index];
		const node = document.createElement('div');
		const [ startX, startY, endX, endY ] = face.box;

		node.className = 'face-box';
		node.dataset.personId = person ? person.id : '';
		node.style.left = (startX * 100) + '%';
		node.style.top = (startY * 100) + '%';
		node.style.width = ((endX - startX) * 100) + '%';
		node.style.height = ((endY - startY) * 100) + '%';
		node.classList.toggle('is-assigned', Boolean(person && person.source_id));
		node.classList.toggle('is-unknown', !person);
		node.classList.toggle('is-hover', Boolean(person && person.id === state.hoverPersonId));
		node.title = person ? t('person.name', { number: index + 1 }) : t('face.unknown');
		node.innerHTML = '<span class="face-badge">' + (person ? index + 1 : '+') + '</span>';

		if (person && person.source_thumbnail)
		{
			const avatar = document.createElement('span');
			avatar.className = 'face-avatar';
			avatar.style.backgroundImage = 'url(' + person.source_thumbnail + ')';
			node.appendChild(avatar);
		}
		node.addEventListener('mouseenter', () => setHoverPerson(person ? person.id : null));
		node.addEventListener('mouseleave', () => setHoverPerson(null));
		node.addEventListener('click', () => onFaceClick(person));
		fragment.appendChild(node);
	});
	el.faceLayer.replaceChildren(fragment);
	renderStickerLayer();
}

function isLiveSticker()
{
	const person = people().find(item => item.id === state.adjustPersonId);
	return state.mode === 'result' && Boolean(person && person.source_kind === 'sticker');
}

function renderStickerLayer()
{
	// while adjusting, the engine preview comes without stickers and they are drawn here, live
	const live = isLiveSticker();
	const fragment = document.createDocumentFragment();

	el.stickerLayer.hidden = !live;

	if (live)
	{
		state.faces.forEach(face =>
		{
			const person = people()[personIndex(face.person_id)];

			if (person && person.source_kind === 'sticker' && face.anchor)
			{
				fragment.appendChild(createStickerOverlay(person, face.anchor));
			}
		});
	}
	el.stickerLayer.replaceChildren(fragment);
}

function createStickerOverlay(person, anchor)
{
	// mirrors paste_sticker() in the face_sticker processor, in percent of the frame so resizes keep it in place
	const frameWidth = target().width;
	const frameHeight = target().height;
	const options = person.sticker;
	const angle = (options.follow_rotation ? anchor.angle : 0) + options.rotation;
	const radians = angle * Math.PI / 180;
	const faceSize = anchor.size * frameWidth;
	const offsetX = options.offset_x * faceSize;
	const offsetY = options.offset_y * faceSize;
	const centerX = anchor.x * frameWidth + offsetX * Math.cos(radians) - offsetY * Math.sin(radians);
	const centerY = anchor.y * frameHeight + offsetX * Math.sin(radians) + offsetY * Math.cos(radians);
	const [ stickerWidth, stickerHeight ] = person.sticker_size;
	const stickerScale = faceSize * options.scale / Math.max(stickerWidth, stickerHeight);
	const width = stickerWidth * stickerScale;
	const height = stickerHeight * stickerScale;
	const node = document.createElement('img');

	node.className = 'face-sticker';
	node.src = person.sticker_image;
	node.alt = '';
	node.style.left = ((centerX - width / 2) / frameWidth * 100) + '%';
	node.style.top = ((centerY - height / 2) / frameHeight * 100) + '%';
	node.style.width = (width / frameWidth * 100) + '%';
	node.style.height = (height / frameHeight * 100) + '%';
	// css 3d mirrors project_sticker_corners(): perspective, then rotateZ, rotateX, rotateY applied right to left
	const depth = options.depth || 0;
	const displayScale = el.frame.clientWidth / frameWidth;
	const perspective = 3 * Math.max(width, height) * displayScale;

	node.style.transform = 'perspective(' + perspective + 'px) rotateZ(' + angle + 'deg) rotateX(' + ((anchor.pitch || 0) * depth) + 'deg) rotateY(' + ((anchor.yaw || 0) * depth) + 'deg)' + (options.mirror ? ' scaleX(-1)' : '');
	return node;
}

function onFaceClick(person)
{
	if (!person)
	{
		scanFrame();
		return;
	}
	if (state.armedId)
	{
		assignSource(person.id, state.armedId);
		return;
	}
	const row = el.peopleList.querySelector('[data-person-id="' + person.id + '"]');

	if (row)
	{
		row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
		row.classList.remove('is-flash');
		void row.offsetWidth;
		row.classList.add('is-flash');
	}
}

function setHoverPerson(personId)
{
	state.hoverPersonId = personId;
	el.faceLayer.querySelectorAll('.face-box').forEach(node => node.classList.toggle('is-hover', personId !== null && Number(node.dataset.personId) === personId));
	el.peopleList.querySelectorAll('.person').forEach(node => node.classList.toggle('is-hover', personId !== null && Number(node.dataset.personId) === personId));
}

function renderTrim()
{
	const [ trimStart, trimEnd ] = state.session.trim;
	const total = Math.max(1, target().frame_total - 1);
	const startPercent = ((trimStart || 0) / total) * 100;
	const endPercent = ((trimEnd ? trimEnd - 1 : total) / total) * 100;
	const hasTrim = trimStart !== null || trimEnd !== null;

	el.scrubberTrim.style.background = hasTrim
		? 'linear-gradient(to right, var(--fill) ' + startPercent + '%, var(--accent) ' + startPercent + '%, var(--accent) ' + endPercent + '%, var(--fill) ' + endPercent + '%)'
		: '';
	el.trimStart.classList.toggle('is-set', trimStart !== null);
	el.trimEnd.classList.toggle('is-set', trimEnd !== null);
	el.trimReset.hidden = !hasTrim;
}

async function updateTrim(start, end)
{
	try
	{
		setSession(await request('POST', '/api/trim', { start, end }));
	}
	catch (error)
	{
		handleError(error);
	}
}

/* ------------------------------------------------------------ people */

function renderPeople()
{
	const list = people();
	const count = assignedCount();
	const fragment = document.createDocumentFragment();
	const renderedIds = new Set();

	el.peopleCount.textContent = list.length;

	if (!target())
	{
		el.peopleSubtitle.textContent = t('people.no_target');
	}
	else if (!list.length)
	{
		el.peopleSubtitle.textContent = isVideo() ? t('people.none_video') : t('people.none_image');
	}
	else
	{
		el.peopleSubtitle.textContent = t('people.summary', { count, total: list.length });
	}

	list.forEach((person, index) =>
	{
		const node = document.createElement('li');
		const isNew = !state.renderedPersonIds.has(person.id);
		const appearance = isVideo() ? t('person.seen', { count: person.count }) : t('person.on_photo');
		const name = escapeHtml(t('person.name', { number: index + 1 }));

		renderedIds.add(person.id);
		node.className = 'person' + (person.source_id ? ' is-assigned' : '') + (person.id === state.hoverPersonId ? ' is-hover' : '');
		node.dataset.personId = person.id;

		if (!isNew)
		{
			node.style.animation = 'none';
		}
		node.innerHTML =
			'<button class="person-target" title="' + (isVideo() ? escapeHtml(t('person.go_to_frame')) : name) + '" style="background-image:url(' + person.thumbnail + ')"><span class="person-number">' + (index + 1) + '</span></button>' +
			ICONS.chevron +
			'<button class="slot' + (person.source_id ? ' is-filled' : '') + (person.source_kind === 'sticker' ? ' is-sticker' : '') + '" title="' + escapeHtml(person.source_id ? t('person.change_source') : t('person.choose_source')) + '"' + (person.source_thumbnail ? ' style="background-image:url(' + person.source_thumbnail + ')"' : '') + '>' +
				(person.source_id ? '<span class="mini-remove" title="' + escapeHtml(t('person.remove_source')) + '">' + ICONS.close + '</span>' : ICONS.plus) +
			'</button>' +
			'<div class="person-meta"><strong>' + name + '</strong><span>' + escapeHtml(appearance + ' · ' + describePerson(person)) + '</span></div>' +
			'<div class="person-actions">' +
				(person.source_kind === 'sticker' ? '<button class="icon-button person-adjust' + (state.adjustPersonId === person.id ? ' is-active' : '') + '" title="' + escapeHtml(t('person.adjust')) + '">' + ICONS.sliders + '</button>' : '') +
				'<button class="icon-button person-remove" title="' + escapeHtml(t('person.remove')) + '">' + ICONS.close + '</button>' +
			'</div>';

		if (person.source_kind === 'sticker' && state.adjustPersonId === person.id)
		{
			node.appendChild(buildAdjustPanel(person));
		}

		const targetButton = node.querySelector('.person-target');
		const slot = node.querySelector('.slot');
		const removeSource = slot.querySelector('.mini-remove');

		targetButton.addEventListener('click', () => isVideo() && goToFrame(person.frame_number));
		slot.addEventListener('click', () =>
		{
			if (state.armedId)
			{
				assignSource(person.id, state.armedId);
			}
			else
			{
				state.slotPersonId = person.id;
				el.slotInput.value = '';
				el.slotInput.click();
			}
		});

		if (removeSource)
		{
			removeSource.addEventListener('click', event =>
			{
				event.stopPropagation();
				assignSource(person.id, null);
			});
		}
		makeDropZone(slot,
		{
			onLibrary: sourceId => assignSource(person.id, sourceId),
			onFiles: files => uploadAndAssign(person.id, files)
		});
		node.querySelector('.person-remove').addEventListener('click', () => removePerson(person.id));

		if (node.querySelector('.person-adjust'))
		{
			node.querySelector('.person-adjust').addEventListener('click', () => toggleAdjust(person));
		}
		node.addEventListener('mouseenter', () => setHoverPerson(person.id));
		node.addEventListener('mouseleave', () => setHoverPerson(null));
		fragment.appendChild(node);
	});

	state.renderedPersonIds = renderedIds;
	el.peopleList.replaceChildren(fragment);
	el.peopleEmpty.hidden = list.length > 0;
	el.peopleFooter.hidden = list.length === 0;
}

function describePerson(person)
{
	if (person.source_kind === 'sticker')
	{
		return t('person.sticker');
	}
	return person.source_id ? t('person.swapped') : t('person.kept');
}

function toggleAdjust(person)
{
	state.adjustPersonId = state.adjustPersonId === person.id ? null : person.id;
	state.previewDirty = true;

	// show the person where they are best visible while adjusting
	if (state.adjustPersonId && isVideo())
	{
		goToFrame(person.frame_number);
	}
	renderPeople();

	if (state.adjustPersonId && state.mode !== 'result')
	{
		setMode('result');
	}
	else
	{
		renderMode();
		invalidatePreview();
	}
}

const STICKER_SLIDERS =
[
	{ key: 'scale', label: 'adjust.scale', min: 0.2, max: 5, step: 0.05, format: value => formatPercent(Math.round(value * 100)) },
	{ key: 'offset_x', label: 'adjust.offset_x', min: -1, max: 1, step: 0.01, format: value => (value > 0 ? '+' : '') + Math.round(value * 100) },
	{ key: 'offset_y', label: 'adjust.offset_y', min: -1, max: 1, step: 0.01, format: value => (value > 0 ? '+' : '') + Math.round(-value * 100) },
	{ key: 'rotation', label: 'adjust.rotation', min: -180, max: 180, step: 1, format: value => Math.round(value) + '°' },
	{ key: 'depth', label: 'adjust.depth', min: 0, max: 1, step: 0.05, format: value => formatPercent(Math.round(value * 100)) }
];

const STICKER_TOGGLES =
[
	{ key: 'follow_rotation', label: 'adjust.follow_rotation' },
	{ key: 'mirror', label: 'adjust.mirror' }
];

const STICKER_SHAPES =
[
	{ value: 'full', label: 'adjust.shape_full', round: false, cutout: false },
	{ value: 'round', label: 'adjust.shape_round', round: true, cutout: false },
	{ value: 'cutout', label: 'adjust.shape_cutout', round: false, cutout: true }
];

function stickerShape(sticker)
{
	if (sticker.cutout)
	{
		return 'cutout';
	}
	return sticker.round ? 'round' : 'full';
}

function buildAdjustPanel(person)
{
	const panel = document.createElement('div');

	panel.className = 'adjust';
	STICKER_SLIDERS.forEach(slider =>
	{
		const row = document.createElement('label');
		const value = person.sticker[slider.key] ?? 0;
		const shownValue = slider.key === 'offset_y' ? -value : value;

		row.className = 'adjust-row';
		row.innerHTML = '<span>' + escapeHtml(t(slider.label)) + '</span><input type="range" min="' + slider.min + '" max="' + slider.max + '" step="' + slider.step + '" value="' + shownValue + '"><output>' + slider.format(value) + '</output>';
		const input = row.querySelector('input');
		const output = row.querySelector('output');

		input.addEventListener('input', () =>
		{
			// the vertical slider reads "up" as positive, the engine offset points down
			const nextValue = slider.key === 'offset_y' ? -Number(input.value) : Number(input.value);

			person.sticker[slider.key] = nextValue;
			output.textContent = slider.format(nextValue);
			renderStickerLayer();
			saveSticker(person);
		});
		panel.appendChild(row);
	});

	const shapes = document.createElement('div');

	shapes.className = 'adjust-shapes';
	shapes.innerHTML = '<span>' + escapeHtml(t('adjust.shape')) + '</span>';
	STICKER_SHAPES.forEach(shape =>
	{
		const chip = document.createElement('button');

		chip.className = 'chip' + (stickerShape(person.sticker) === shape.value ? ' is-set' : '');
		chip.textContent = t(shape.label);
		chip.addEventListener('click', async () =>
		{
			shapes.querySelectorAll('.chip').forEach(node => node.classList.toggle('is-set', node === chip));
			person.sticker.round = shape.round;
			person.sticker.cutout = shape.cutout;

			if (shape.cutout)
			{
				chip.textContent = t('adjust.cutting_out');
			}
			renderStickerLayer();
			await saveSticker(person, true);
			chip.textContent = t(shape.label);
		});
		shapes.appendChild(chip);
	});
	panel.appendChild(shapes);

	const toggles = document.createElement('div');

	toggles.className = 'adjust-toggles';
	STICKER_TOGGLES.forEach(toggle =>
	{
		const chip = document.createElement('button');

		chip.className = 'chip' + (person.sticker[toggle.key] ? ' is-set' : '');
		chip.textContent = t(toggle.label);
		chip.addEventListener('click', () =>
		{
			person.sticker[toggle.key] = !person.sticker[toggle.key];
			chip.classList.toggle('is-set', person.sticker[toggle.key]);
			renderStickerLayer();
			saveSticker(person, true);
		});
		toggles.appendChild(chip);
	});

	const reset = document.createElement('button');

	reset.className = 'link-button';
	reset.textContent = t('adjust.reset');
	reset.addEventListener('click', async () =>
	{
		try
		{
			setSession(await request('POST', '/api/people/' + person.id + '/sticker', { reset: true }));
			invalidatePreview();
		}
		catch (error)
		{
			handleError(error);
		}
	});
	toggles.appendChild(reset);
	panel.appendChild(toggles);
	return panel;
}

function saveSticker(person, immediate)
{
	clearTimeout(state.stickerTimers[person.id]);
	return new Promise(resolve =>
	{
		state.stickerTimers[person.id] = setTimeout(async () =>
		{
			try
			{
				const session = await withDownloadWatch(() => request('POST', '/api/people/' + person.id + '/sticker', person.sticker));

				// keep the open panel (and the slider being dragged) instead of re-rendering the list
				state.session = session;
				renderToolbar();
				renderFaces();

				// the live overlay already shows the change, the baked preview is refreshed once adjusting ends
				if (isLiveSticker())
				{
					state.previewDirty = true;
				}
				else
				{
					invalidatePreview();
				}
			}
			catch (error)
			{
				handleError(error);
				// the server refused (for instance the cut-out failed), show its state again
				setSession(await request('GET', '/api/state').catch(() => state.session));
			}
			resolve();
		}, immediate ? 0 : 250);
	});
}

async function assignSource(personId, sourceId)
{
	if (sourceId && sourceId === state.armedId)
	{
		setArmed(null);
	}
	try
	{
		setSession(await request('POST', '/api/people/' + personId + '/source', { source_id: sourceId }));
		invalidatePreview();
	}
	catch (error)
	{
		handleError(error);
	}
}

async function uploadAndAssign(personId, files)
{
	const images = Array.from(files).filter(file => file.type.startsWith('image/'));

	if (!images.length)
	{
		toast(t('toast.pick_image'), 'error');
		return;
	}
	try
	{
		const session = await upload('/api/library?kind=' + state.libraryKind, images.slice(0, 1), 'files');

		reportRejected(session.rejected);
		setSession(session);

		if (session.added.length)
		{
			await assignSource(personId, session.added[0]);
		}
	}
	catch (error)
	{
		handleError(error);
	}
}

async function removePerson(personId)
{
	try
	{
		setSession(await request('DELETE', '/api/people/' + personId));
		invalidatePreview();
		fetchFaces();
	}
	catch (error)
	{
		handleError(error);
	}
}

async function scanFrame()
{
	if (!target())
	{
		return;
	}
	try
	{
		el.scanFrame.disabled = true;
		const session = await request('POST', '/api/scan', { mode: 'frame', frame_number: state.frameNumber });

		setSession(session);
		toast(session.created ? tc('toast.created', session.created) : t('toast.created_none'), session.created ? 'success' : 'info');
		fetchFaces();
	}
	catch (error)
	{
		handleError(error);
	}
	finally
	{
		el.scanFrame.disabled = !target() || isBusy();
	}
}

async function scanVideo()
{
	try
	{
		setSession(await request('POST', '/api/scan', { mode: 'video', samples: 60 }));
		startPolling();
	}
	catch (error)
	{
		handleError(error);
	}
}

/* ------------------------------------------------------------ library */

function renderLibrary()
{
	const items = library().filter(item => (item.kind || 'face') === state.libraryKind);
	const fragment = document.createDocumentFragment();

	if (state.armedId && !library().some(item => item.id === state.armedId))
	{
		setArmed(null);
	}
	renderLibraryKind();

	if (!items.length)
	{
		const empty = document.createElement('div');
		empty.className = 'library-empty';
		empty.textContent = t(LIBRARY_KINDS[state.libraryKind].empty);
		fragment.appendChild(empty);
	}
	items.forEach(item =>
	{
		const node = document.createElement('button');

		node.className = 'library-item' + (item.kind === 'sticker' ? ' is-sticker' : '') + (item.id === state.armedId ? ' is-armed' : '');
		node.dataset.id = item.id;
		node.draggable = true;
		node.title = t('library.item_title', { name: item.name });
		node.innerHTML = '<img alt="" src="' + item.thumbnail + '"><span class="item-name">' + escapeHtml(item.name.replace(/\.[^.]+$/, '')) + '</span><span class="mini-remove remove" title="' + escapeHtml(t('library.remove')) + '">' + ICONS.close + '</span>';
		node.addEventListener('click', () => setArmed(state.armedId === item.id ? null : item.id));
		node.addEventListener('dragstart', event =>
		{
			event.dataTransfer.setData(LIBRARY_MIME, item.id);
			event.dataTransfer.setData('text/plain', item.name);
			event.dataTransfer.effectAllowed = 'copy';
		});
		node.querySelector('.remove').addEventListener('click', event =>
		{
			event.stopPropagation();
			removeLibraryItem(item.id);
		});
		fragment.appendChild(node);
	});
	el.librarySt.replaceChildren(fragment);
	updateStripFade();
}

function updateStripFade()
{
	const strip = el.librarySt;

	strip.classList.toggle('has-more-left', strip.scrollLeft > 4);
	strip.classList.toggle('has-more-right', strip.scrollLeft + strip.clientWidth < strip.scrollWidth - 4);
}

function installStripScroll()
{
	// a mouse wheel only scrolls vertically, turn it into horizontal scrolling for the source strip
	el.librarySt.addEventListener('wheel', event =>
	{
		if (Math.abs(event.deltaY) > Math.abs(event.deltaX) && el.librarySt.scrollWidth > el.librarySt.clientWidth)
		{
			el.librarySt.scrollLeft += event.deltaY;
			event.preventDefault();
			updateStripFade();
		}
	}, { passive: false });
	el.librarySt.addEventListener('scroll', updateStripFade, { passive: true });
	new ResizeObserver(updateStripFade).observe(el.librarySt);
}

function setArmed(sourceId)
{
	state.armedId = sourceId;
	document.body.classList.toggle('is-armed', Boolean(sourceId));
	el.librarySt.querySelectorAll('.library-item').forEach(node => node.classList.toggle('is-armed', node.dataset.id === sourceId));
	renderMode();
}

function renderLibraryKind()
{
	const buttons = Array.from(el.libraryKind.querySelectorAll('button'));
	const activeButton = buttons.find(button => button.dataset.kind === state.libraryKind);

	buttons.forEach(button => button.classList.toggle('is-active', button === activeButton));
	placeSegmentThumb(el.libraryKind, activeButton);
	el.libraryHint.textContent = t(LIBRARY_KINDS[state.libraryKind].hint);
}

function setLibraryKind(kind)
{
	state.libraryKind = kind;
	renderLibrary();
}

async function addSources(files)
{
	const images = Array.from(files).filter(file => file.type.startsWith('image/'));
	const kind = state.libraryKind;

	if (!images.length)
	{
		return;
	}
	try
	{
		toast(tc('library.analysing', images.length), 'info', 1800);
		const session = await upload('/api/library?kind=' + kind, images, 'files');

		reportRejected(session.rejected, kind);
		setSession(session);

		if (session.added.length)
		{
			toast(tc(LIBRARY_KINDS[kind].added, session.added.length));
		}
	}
	catch (error)
	{
		handleError(error);
	}
}

function reportRejected(rejected, kind = state.libraryKind)
{
	if (rejected && rejected.length)
	{
		toast(t(LIBRARY_KINDS[kind].rejected, { names: rejected.join(', ') }), 'error', 5200);
	}
}

async function removeLibraryItem(sourceId)
{
	try
	{
		setSession(await request('DELETE', '/api/library/' + sourceId));
		invalidatePreview();
	}
	catch (error)
	{
		handleError(error);
	}
}

/* ------------------------------------------------------------ drag & drop */

function dragKind(event)
{
	const types = Array.from(event.dataTransfer ? event.dataTransfer.types : []);

	if (types.includes(LIBRARY_MIME))
	{
		return 'library';
	}
	if (types.includes('Files'))
	{
		return 'files';
	}
	return null;
}

function makeDropZone(node, handlers)
{
	let depth = 0;

	node.addEventListener('dragenter', event =>
	{
		const kind = dragKind(event);

		if ((kind === 'library' && handlers.onLibrary) || (kind === 'files' && handlers.onFiles))
		{
			event.preventDefault();
			event.stopPropagation();
			depth++;
			node.classList.add('is-dragover');
		}
	});
	node.addEventListener('dragover', event =>
	{
		const kind = dragKind(event);

		if ((kind === 'library' && handlers.onLibrary) || (kind === 'files' && handlers.onFiles))
		{
			event.preventDefault();
			event.stopPropagation();
			event.dataTransfer.dropEffect = 'copy';
		}
	});
	node.addEventListener('dragleave', () =>
	{
		depth = Math.max(0, depth - 1);

		if (!depth)
		{
			node.classList.remove('is-dragover');
		}
	});
	node.addEventListener('drop', event =>
	{
		const kind = dragKind(event);

		depth = 0;
		node.classList.remove('is-dragover');

		if (kind === 'library' && handlers.onLibrary)
		{
			event.preventDefault();
			event.stopPropagation();
			handlers.onLibrary(event.dataTransfer.getData(LIBRARY_MIME));
		}
		if (kind === 'files' && handlers.onFiles)
		{
			event.preventDefault();
			event.stopPropagation();
			handlers.onFiles(event.dataTransfer.files);
		}
	});
}

function installGlobalDrop()
{
	let depth = 0;
	const endDrag = () =>
	{
		depth = 0;
		el.dropOverlay.hidden = true;
		el.canvasDrop.hidden = true;
		document.body.classList.remove('is-dragging');
	};

	// counted while capturing: the drop zones stop these events before they would bubble up here
	window.addEventListener('dragenter', event =>
	{
		if (dragKind(event) === 'files' && !depth++)
		{
			el.dropOverlayText.textContent = t('drop.open');
			el.dropOverlay.hidden = Boolean(target());
			el.canvasDropText.textContent = t(isVideo() ? 'drop.replace_video' : 'drop.replace_image');
			el.canvasDrop.hidden = !target() || isBusy();
			document.body.classList.add('is-dragging');
		}
	}, true);
	window.addEventListener('dragleave', event =>
	{
		if (dragKind(event) === 'files' && depth && !--depth)
		{
			endDrag();
		}
	}, true);
	window.addEventListener('drop', endDrag, true);
	window.addEventListener('dragover', event => event.preventDefault());
	window.addEventListener('drop', event =>
	{
		event.preventDefault();

		if (dragKind(event) === 'files' && !target() && event.dataTransfer.files.length)
		{
			openTarget(event.dataTransfer.files[0]);
		}
	});
	makeDropZone(el.canvas, { onFiles: files => files.length && openTarget(files[0]) });
	makeDropZone(el.library, { onFiles: addSources });
}

/* ------------------------------------------------------------ target */

async function openTarget(file)
{
	if (isBusy() || state.importing)
	{
		return;
	}
	if (!file.type.startsWith('image/') && !file.type.startsWith('video/'))
	{
		toast(t('toast.pick_media'), 'error');
		return;
	}
	// replacing the target resets the people, so assignments are only dropped on purpose
	if (assignedCount() && !await confirmDialog(t(isVideo() ? 'replace.title_video' : 'replace.title_image'), t('replace.message', { name: file.name })))
	{
		return;
	}
	state.importing = true;
	showHud({ stage: t('import.stage'), message: file.name, progress: 0 }, false);
	try
	{
		const session = await upload('/api/target', [ file ], 'file', progress =>
		{
			showHud(progress < 1 ? { stage: t('import.stage'), message: file.name, progress: progress * 0.8 } : { stage: t('import.detecting'), message: t('import.wait'), progress: 0.9 }, false);
		});

		state.frameNumber = 0;
		state.mode = 'original';
		state.faces = [];
		state.renderedPersonIds = new Set();
		setSession(session);
		showFrame();

		if (isVideo())
		{
			await scanVideo();
		}
		else
		{
			toast(people().length ? tc('toast.detected', people().length) : t('toast.detected_none'), people().length ? 'success' : 'info');
		}
	}
	catch (error)
	{
		handleError(error);
	}
	finally
	{
		state.importing = false;

		if (!isBusy())
		{
			hideHud();
		}
	}
}

function confirmDialog(title, message)
{
	return new Promise(resolve =>
	{
		const close = answer =>
		{
			el.confirm.hidden = true;
			el.confirmAccept.removeEventListener('click', accept);
			el.confirmCancel.removeEventListener('click', cancel);
			el.confirm.removeEventListener('click', backdrop);
			document.removeEventListener('keydown', keydown, true);
			resolve(answer);
		};
		const accept = () => close(true);
		const cancel = () => close(false);
		const backdrop = event => event.target === el.confirm && close(false);
		const keydown = event =>
		{
			if (event.key === 'Escape')
			{
				event.stopPropagation();
				close(false);
			}
		};

		el.confirmTitle.textContent = title;
		el.confirmMessage.textContent = message;
		el.confirmAccept.addEventListener('click', accept);
		el.confirmCancel.addEventListener('click', cancel);
		el.confirm.addEventListener('click', backdrop);
		document.addEventListener('keydown', keydown, true);
		el.confirm.hidden = false;
		el.confirmAccept.focus();
	});
}

/* ------------------------------------------------------------ tasks */

function renderTask()
{
	const task = state.session && state.session.task;
	const running = isBusy();

	if (running && task.kind === 'scan')
	{
		el.scanProgress.hidden = false;
		el.scanBar.style.width = Math.round(task.progress * 100) + '%';
		el.scanText.textContent = translateServer(task.stage || 'stage.scan') + ' · ' + Math.round(task.progress * 100) + '%' + (task.message ? ' · ' + translateServer(task.message, task.message_params) : '');
	}
	else
	{
		el.scanProgress.hidden = true;
	}

	if (running && [ 'render', 'setup' ].includes(task.kind))
	{
		showHud(task, task.kind === 'render');
	}
	else if (!state.importing)
	{
		hideHud();
	}
}

function showHud(task, cancellable)
{
	const download = task.download;
	const progress = Math.max(0, Math.min(1, download ? download.progress : task.progress || 0));
	const indeterminate = task.kind === 'setup' && !download;

	el.hud.hidden = false;
	el.hudStage.textContent = download ? t('hud.download') : translateServer(task.stage) || t('hud.preparing');
	el.hudMessage.textContent = download ? download.name + ' · ' + formatBytes(download.received) + ' / ' + formatBytes(download.total) : translateServer(task.message, task.message_params) || (indeterminate ? t('hud.checking_models') : '');
	el.hudPercent.textContent = indeterminate ? '' : Math.round(progress * 100) + '%';
	el.hudRing.style.strokeDashoffset = String(326.73 * (1 - (indeterminate ? 0.25 : progress)));
	el.hudRingBox.classList.toggle('is-indeterminate', indeterminate);
	el.hudCancel.hidden = !cancellable;
}

function formatBytes(bytes)
{
	if (bytes >= 1024 * 1024 * 1024)
	{
		return t('unit.gigabytes', { value: formatDecimal(bytes / (1024 * 1024 * 1024), 1) });
	}
	return t('unit.megabytes', { value: Math.round(bytes / (1024 * 1024)) });
}

function renderDownload(download)
{
	el.downloadCard.hidden = !download;

	if (download)
	{
		const percent = Math.round(download.progress * 100);

		el.downloadName.textContent = download.name;
		el.downloadPercent.textContent = formatPercent(percent);
		el.downloadBar.style.width = percent + '%';
		el.downloadSize.textContent = t('download.size', { received: formatBytes(download.received), total: formatBytes(download.total) });
	}
}

async function withDownloadWatch(work)
{
	// slow requests may be fetching a model, show its progress while they run
	const poll = async () =>
	{
		try
		{
			renderDownload((await request('GET', '/api/download')).download);
		}
		catch (error)
		{
			/* the card is optional */
		}
	};
	const timer = setInterval(poll, 600);

	try
	{
		return await work();
	}
	finally
	{
		clearInterval(timer);
		renderDownload(null);
	}
}

function hideHud()
{
	el.hud.hidden = true;
}

function startPolling()
{
	if (state.pollTimer)
	{
		return;
	}
	state.pollTimer = setInterval(pollTask, 700);
}

async function pollTask()
{
	try
	{
		const task = state.session.task;
		const session = task && task.kind === 'scan' ? await request('GET', '/api/state') : Object.assign({}, state.session, { task: await request('GET', '/api/task') });

		if (session.task.status !== 'running')
		{
			clearInterval(state.pollTimer);
			state.pollTimer = null;
			setSession(await request('GET', '/api/state'));
			onTaskFinished(session.task);
			return;
		}
		setSession(session);
	}
	catch (error)
	{
		clearInterval(state.pollTimer);
		state.pollTimer = null;
		handleError(error);
	}
}

function onTaskFinished(task)
{
	if (!task.id || task.id === state.lastHandledTaskId)
	{
		return;
	}
	state.lastHandledTaskId = task.id;

	if (task.status === 'error')
	{
		toast(translateServer(task.message, task.message_params), 'error', 6500);
	}
	if (task.status === 'done' && task.kind === 'setup')
	{
		state.options = null;
		request('GET', '/api/options').then(options => { state.options = options; }).catch(() => {});
	}
	if (task.status === 'cancelled')
	{
		toast(t('toast.cancelled'), 'info');
	}
	if (task.status === 'done' && task.kind === 'scan')
	{
		toast(people().length ? tc('toast.found', people().length) : t('toast.found_none'), people().length ? 'success' : 'info');
		fetchFaces();
		invalidatePreview();
	}
	if (task.status === 'done' && task.kind === 'render' && task.result && task.result.url)
	{
		showResult(task.result);
	}
}

async function startRender()
{
	try
	{
		setSession(await request('POST', '/api/render'));
		startPolling();
	}
	catch (error)
	{
		handleError(error);
	}
}

async function cancelTask()
{
	try
	{
		await request('POST', '/api/task/cancel');
	}
	catch (error)
	{
		handleError(error);
	}
}

function showResult(output)
{
	const url = '/api/output/' + encodeURIComponent(output.name);
	const media = document.createElement(output.type === 'video' ? 'video' : 'img');

	media.src = url + '?t=' + Date.now();

	if (output.type === 'video')
	{
		Object.assign(media, { controls: true, autoplay: true, loop: true, playsInline: true });
	}
	else
	{
		media.alt = t('result.alt');
	}
	el.resultName.textContent = output.name;
	el.revealLabel.textContent = state.options && state.options.platform === 'darwin' ? t('result.reveal_finder') : t('result.reveal_folder');
	el.resultMedia.replaceChildren(media);
	el.downloadResult.href = url;
	el.downloadResult.setAttribute('download', output.name);
	el.result.hidden = false;
}

function closeResult()
{
	const video = el.resultMedia.querySelector('video');

	if (video)
	{
		video.pause();
	}
	el.result.hidden = true;
	el.resultMedia.replaceChildren();
}

/* ------------------------------------------------------------ settings */

const PROVIDER_LABELS =
{
	coreml: 'Apple Neural Engine',
	cuda: 'NVIDIA CUDA',
	tensorrt: 'NVIDIA TensorRT',
	directml: 'DirectML',
	openvino: 'OpenVINO',
	rocm: 'AMD ROCm',
	migraphx: 'AMD MIGraphX'
};

function settingsSchema()
{
	const settings = state.session.settings;
	const options = state.options;
	const pixelBoosts = options.face_swapper_pixel_boosts[settings.face_swapper_model] || [];

	return [
		{
			title: t('settings.swap'),
			rows:
			[
				{ type: 'select', key: 'face_swapper_model', label: t('settings.model'), choices: options.face_swapper_models.map(model => ({ value: model.value, label: model.label + (model.downloaded ? '' : t('settings.to_download')) })) },
				{ type: 'select', key: 'face_swapper_pixel_boost', label: t('settings.pixel_boost'), choices: pixelBoosts.map(value => ({ value, label: value.split('x')[0] + ' px' })) },
				{ type: 'slider', key: 'face_swapper_weight', label: t('settings.weight'), min: 0, max: 1, step: 0.05, format: value => formatPercent(Math.round(value * 100)) }
			],
			footnote: t('settings.swap_note')
		},
		{
			title: t('settings.enhance'),
			rows:
			[
				{ type: 'switch', key: 'face_enhancer', label: t('settings.enhancer') },
				{ type: 'select', key: 'face_enhancer_model', label: t('settings.model'), hidden: !settings.face_enhancer, choices: options.face_enhancer_models.map(model => ({ value: model.value, label: model.label + (model.downloaded ? '' : t('settings.to_download')) })) },
				{ type: 'slider', key: 'face_enhancer_blend', label: t('settings.blend'), hidden: !settings.face_enhancer, min: 0, max: 100, step: 1, format: value => formatPercent(value) }
			]
		},
		{
			title: t('settings.mask'),
			rows:
			[
				{ type: 'switch', key: 'occlusion', label: t('settings.occlusion') },
				{ type: 'slider', key: 'face_mask_blur', label: t('settings.mask_blur'), min: 0, max: 1, step: 0.05, format: value => formatPercent(Math.round(value * 100)) }
			]
		},
		{
			title: t('settings.recognition'),
			rows:
			[
				{ type: 'slider', key: 'reference_face_distance', label: t('settings.distance'), min: 0.05, max: 0.95, step: 0.05, format: value => formatPercent(Math.round(value * 100)) },
				{ type: 'select', key: 'face_detector_model', label: t('settings.detector'), choices: options.face_detector_models.map(value => ({ value, label: value === 'many' ? t('settings.detector_all') : value })) },
				{ type: 'slider', key: 'face_detector_score', label: t('settings.detector_score'), min: 0.05, max: 0.95, step: 0.05, format: value => formatPercent(Math.round(value * 100)) }
			],
			footnote: t('settings.recognition_note')
		},
		{
			title: t('settings.export'),
			rows:
			[
				{ type: 'slider', key: 'output_image_quality', label: t('settings.image_quality'), min: 10, max: 100, step: 1, format: value => formatPercent(value) },
				{ type: 'slider', key: 'output_video_quality', label: t('settings.video_quality'), min: 10, max: 100, step: 1, format: value => formatPercent(value) }
			]
		},
		{
			title: t('settings.performance'),
			rows:
			[
				{ type: 'select', key: 'execution_provider', label: t('settings.provider'), choices: options.execution_providers.map(value => ({ value, label: value === 'cpu' ? t('settings.provider_cpu') : PROVIDER_LABELS[value] || value })) }
			]
		}
	];
}

function buildSettings()
{
	const settings = state.session.settings;
	const fragment = document.createDocumentFragment();

	settingsSchema().forEach(group =>
	{
		const title = document.createElement('div');
		const box = document.createElement('div');

		title.className = 'group-title';
		title.textContent = group.title;
		box.className = 'group';
		group.rows.filter(row => !row.hidden).forEach(row => box.appendChild(buildSettingRow(row, settings[row.key])));
		fragment.appendChild(title);
		fragment.appendChild(box);

		if (group.footnote)
		{
			const footnote = document.createElement('p');
			footnote.className = 'group-footnote';
			footnote.textContent = group.footnote;
			fragment.appendChild(footnote);
		}
	});
	fragment.appendChild(buildAppearanceGroup());
	el.settingsBody.replaceChildren(fragment);
}

function buildSettingRow(row, value)
{
	const node = document.createElement('div');
	const id = 'setting-' + row.key;

	node.className = 'row' + (row.type === 'slider' ? ' row-stack' : '');

	if (row.type === 'select')
	{
		node.innerHTML = '<label for="' + id + '">' + row.label + '</label><span class="select"><select id="' + id + '"></select></span>';
		const select = node.querySelector('select');

		row.choices.forEach(choice => select.add(new Option(choice.label, choice.value, false, choice.value === value)));
		select.addEventListener('change', () => saveSetting(row.key, select.value, true));
	}
	if (row.type === 'switch')
	{
		node.innerHTML = '<label for="' + id + '">' + row.label + '</label><span class="switch"><input type="checkbox" id="' + id + '"' + (value ? ' checked' : '') + '><span></span></span>';
		const input = node.querySelector('input');

		input.addEventListener('change', () => saveSetting(row.key, input.checked, true));
	}
	if (row.type === 'slider')
	{
		node.innerHTML = '<div class="row-line"><label for="' + id + '">' + row.label + '</label><output></output></div><input type="range" id="' + id + '" min="' + row.min + '" max="' + row.max + '" step="' + row.step + '" value="' + value + '">';
		const input = node.querySelector('input');
		const output = node.querySelector('output');
		const update = () =>
		{
			const number = Number(input.value);
			output.textContent = row.format(number);
			input.style.setProperty('--fill-percent', ((number - row.min) / (row.max - row.min) * 100) + '%');
		};

		update();
		input.addEventListener('input', () =>
		{
			update();
			saveSetting(row.key, Number(input.value), false);
		});
	}
	return node;
}

function buildAppearanceGroup()
{
	const wrapper = document.createElement('div');

	wrapper.innerHTML =
		'<div class="group-title">' + escapeHtml(t('settings.appearance')) + '</div>' +
		'<div class="group">' +
		'<div class="row"><label for="setting-language">' + escapeHtml(t('settings.language')) + '</label>' +
		'<span class="select"><select id="setting-language"></select></span></div>' +
		'<div class="row"><label for="setting-theme">' + escapeHtml(t('settings.theme')) + '</label>' +
		'<span class="select"><select id="setting-theme"></select></span></div>' +
		'</div>';
	const languageSelect = wrapper.querySelector('#setting-language');
	const themeSelect = wrapper.querySelector('#setting-theme');

	LANGUAGES.forEach(language => languageSelect.add(new Option(I18N[language]['language.name'], language, false, language === LANGUAGE)));
	[ 'auto', 'light', 'dark' ].forEach(theme => themeSelect.add(new Option(t('settings.theme_' + theme), theme, false, theme === readTheme())));
	languageSelect.addEventListener('change', () => setLanguage(languageSelect.value));
	themeSelect.addEventListener('change', () =>
	{
		writeTheme(themeSelect.value);
		applyTheme();
		requestAnimationFrame(renderMode);
	});
	return wrapper;
}

function setLanguage(language)
{
	writeLanguage(language);
	applyTranslations();
	render();

	if (state.session && state.options)
	{
		buildSettings();
	}
	if (state.faces.length)
	{
		renderFaces();
	}
	// the segmented controls follow the width of their translated labels
	requestAnimationFrame(() =>
	{
		renderMode();
		renderLibraryKind();
	});
}

function saveSetting(key, value, rebuild)
{
	clearTimeout(state.settingsTimers[key]);
	state.settingsTimers[key] = setTimeout(async () =>
	{
		try
		{
			const session = await withDownloadWatch(() => request('POST', '/api/settings', { [key]: value }));

			state.session = session;
			render();

			if (rebuild)
			{
				// the save may have just downloaded a model, so its "to download" mark must go
				state.options = await request('GET', '/api/options').catch(() => state.options);
				buildSettings();
			}
			if ([ 'face_detector_model', 'face_detector_score', 'reference_face_distance' ].includes(key))
			{
				fetchFaces();
			}
			invalidatePreview();
		}
		catch (error)
		{
			handleError(error);
		}
	}, rebuild ? 0 : 280);
}

async function openSettings()
{
	if (!state.options)
	{
		return;
	}
	try
	{
		// models get downloaded by previews and renders, so the "to download" marks must be fresh
		state.options = await request('GET', '/api/options');
	}
	catch (error)
	{
		/* keep the last known options */
	}
	buildSettings();
	el.settings.classList.add('is-open');
	el.settings.setAttribute('aria-hidden', 'false');
	el.settingsBackdrop.hidden = false;
}

function closeSettings()
{
	el.settings.classList.remove('is-open');
	el.settings.setAttribute('aria-hidden', 'true');
	el.settingsBackdrop.hidden = true;
}

function readTheme()
{
	try
	{
		return localStorage.getItem('studio-theme') || 'auto';
	}
	catch (error)
	{
		return 'auto';
	}
}

function writeTheme(theme)
{
	try
	{
		localStorage.setItem('studio-theme', theme);
	}
	catch (error)
	{
		/* private mode: the theme just won't persist */
	}
}

function applyTheme()
{
	const theme = readTheme();

	if (theme === 'auto')
	{
		document.documentElement.removeAttribute('data-theme');
	}
	else
	{
		document.documentElement.setAttribute('data-theme', theme);
	}
}

/* ------------------------------------------------------------ events */

function installCompareHandle()
{
	let dragging = false;
	const move = event =>
	{
		const rect = el.frame.getBoundingClientRect();

		state.comparePosition = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
		updateCompare();
	};

	el.frame.addEventListener('pointerdown', event =>
	{
		if (state.mode === 'compare')
		{
			dragging = true;
			el.frame.setPointerCapture(event.pointerId);
			move(event);
		}
	});
	el.frame.addEventListener('pointermove', event => dragging && move(event));
	el.frame.addEventListener('pointerup', () => { dragging = false; });
	el.frame.addEventListener('pointercancel', () => { dragging = false; });
}

function installEvents()
{
	el.openTarget.addEventListener('click', () => el.targetInput.click());
	el.pickTarget.addEventListener('click', () => el.targetInput.click());
	el.targetInput.addEventListener('change', () =>
	{
		const file = el.targetInput.files[0];

		// cleared so picking the same file again still counts as a change
		el.targetInput.value = '';

		if (file)
		{
			openTarget(file);
		}
	});
	el.addSources.addEventListener('click', () => el.sourcesInput.click());
	el.sourcesInput.addEventListener('change', () =>
	{
		addSources(el.sourcesInput.files);
		el.sourcesInput.value = '';
	});
	el.slotInput.addEventListener('change', () =>
	{
		if (el.slotInput.files.length && state.slotPersonId !== null)
		{
			uploadAndAssign(state.slotPersonId, el.slotInput.files);
		}
	});

	el.modeControl.querySelectorAll('button').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
	el.libraryKind.querySelectorAll('button').forEach(button => button.addEventListener('click', () => setLibraryKind(button.dataset.kind)));
	el.scrubber.addEventListener('input', () =>
	{
		state.frameNumber = Number(el.scrubber.value);
		showFrame();
	});
	el.stepBack.addEventListener('click', () => goToFrame(state.frameNumber - 1));
	el.stepForward.addEventListener('click', () => goToFrame(state.frameNumber + 1));
	el.trimStart.addEventListener('click', () => updateTrim(state.frameNumber, state.session.trim[1] !== null && state.session.trim[1] <= state.frameNumber ? null : state.session.trim[1]));
	el.trimEnd.addEventListener('click', () => updateTrim(state.session.trim[0] !== null && state.session.trim[0] > state.frameNumber ? null : state.session.trim[0], state.frameNumber + 1));
	el.trimReset.addEventListener('click', () => updateTrim(null, null));

	el.scanFrame.addEventListener('click', scanFrame);
	el.scanVideo.addEventListener('click', scanVideo);
	el.scanCancel.addEventListener('click', cancelTask);
	el.assignOrder.addEventListener('click', async () =>
	{
		if (!library().some(item => (item.kind || 'face') === state.libraryKind))
		{
			toast(state.libraryKind === 'sticker' ? t('toast.need_stickers') : t('toast.need_faces'), 'info');
			return;
		}
		try
		{
			setSession(await request('POST', '/api/people/assign-in-order?kind=' + state.libraryKind));
			invalidatePreview();
		}
		catch (error)
		{
			handleError(error);
		}
	});
	el.clearSources.addEventListener('click', async () =>
	{
		try
		{
			setSession(await request('POST', '/api/people/clear-sources'));
			invalidatePreview();
		}
		catch (error)
		{
			handleError(error);
		}
	});

	el.render.addEventListener('click', startRender);
	el.hudCancel.addEventListener('click', cancelTask);
	el.openSettings.addEventListener('click', openSettings);
	el.closeSettings.addEventListener('click', closeSettings);
	el.settingsBackdrop.addEventListener('click', closeSettings);
	el.closeResult.addEventListener('click', closeResult);
	el.result.addEventListener('click', event => event.target === el.result && closeResult());
	el.revealResult.addEventListener('click', () => request('POST', '/api/reveal').catch(handleError));

	document.addEventListener('keydown', event =>
	{
		const typing = [ 'INPUT', 'SELECT', 'TEXTAREA' ].includes(document.activeElement.tagName) && document.activeElement.type !== 'range';

		if (event.key === 'Escape')
		{
			closeSettings();
			closeResult();
			setArmed(null);
		}
		if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'o')
		{
			event.preventDefault();

			if (!isBusy() && !state.importing)
			{
				el.targetInput.click();
			}
		}
		if (!typing && isVideo() && document.activeElement !== el.scrubber)
		{
			if (event.key === 'ArrowLeft')
			{
				goToFrame(state.frameNumber - (event.shiftKey ? 10 : 1));
			}
			if (event.key === 'ArrowRight')
			{
				goToFrame(state.frameNumber + (event.shiftKey ? 10 : 1));
			}
		}
	});

	el.frameOriginal.addEventListener('error', onFrameError);
	el.offlineRetry.addEventListener('click', tryReconnect);
	new ResizeObserver(() => layoutFrame()).observe(el.canvas);
	window.addEventListener('resize', () =>
	{
		renderMode();
		renderLibraryKind();
		renderStickerLayer();
	});
	installCompareHandle();
	installStripScroll();
	installGlobalDrop();
}

/* ------------------------------------------------------------ connection */

async function syncWithServer()
{
	const [ session, options ] = await Promise.all([ request('GET', '/api/state'), request('GET', '/api/options') ]);
	const previousTargetId = target() && target().id;

	state.options = options;

	if (session.task && session.task.status !== 'running')
	{
		state.lastHandledTaskId = session.task.id;
	}
	if (!session.target || session.target.id !== previousTargetId)
	{
		state.frameNumber = 0;
		state.mode = 'original';
		state.faces = [];
		state.renderedPersonIds = new Set();
	}
	setSession(session);

	if (target())
	{
		showFrame();
	}
	if (isBusy())
	{
		startPolling();
	}
	requestAnimationFrame(renderMode);
}

function setOffline(offline)
{
	if (state.offline === offline)
	{
		return;
	}
	state.offline = offline;
	el.offline.hidden = !offline;
	document.body.classList.toggle('is-offline', offline);

	if (offline)
	{
		clearInterval(state.pollTimer);
		state.pollTimer = null;
		state.reconnectTimer = setInterval(tryReconnect, 2000);
	}
	else
	{
		clearInterval(state.reconnectTimer);
		state.reconnectTimer = null;
	}
}

async function tryReconnect()
{
	try
	{
		const response = await fetch('/api/task', { cache: 'no-store' });

		if (!response.ok)
		{
			return;
		}
	}
	catch (error)
	{
		return;
	}
	setOffline(false);

	try
	{
		await syncWithServer();
		toast(t('toast.reconnected'), 'success');
	}
	catch (error)
	{
		handleError(error);
	}
}

function onFrameError()
{
	// a restarted server forgets the target, so a frame that cannot load means our session is stale
	clearTimeout(state.frameErrorTimer);
	state.frameErrorTimer = setTimeout(() =>
	{
		if (!state.offline)
		{
			syncWithServer().catch(() => {});
		}
	}, 400);
}

async function boot()
{
	applyTheme();
	applyTranslations();
	renderToolbar();
	installEvents();

	try
	{
		await syncWithServer();
	}
	catch (error)
	{
		/* setOffline already shows the banner and retries */
	}
	requestAnimationFrame(renderMode);
}

boot();
