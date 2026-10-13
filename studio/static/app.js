'use strict';

const $ = (selector) => document.querySelector(selector);

const el =
{
	targetCaption: $('#target-caption'),
	openTarget: $('#open-target'),
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
	compareHandle: $('#compare-handle'),
	frameSpinner: $('#frame-spinner'),
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
	offline: false,
	reconnectTimer: null,
	frameErrorTimer: null
};

const ICONS =
{
	plus: '<svg viewBox="0 0 24 24"><path d="M12 6v12M6 12h12"/></svg>',
	close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
	chevron: '<svg class="person-arrow" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>'
};

const LIBRARY_MIME = 'application/x-facefusion-source';
const OFFLINE_MESSAGE = 'Le serveur MultiFace Studio est arrêté.';

/* ------------------------------------------------------------ api */

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
		throw new Error(OFFLINE_MESSAGE);
	}
	const data = await response.json().catch(() => ({}));

	if (!response.ok)
	{
		throw new Error(data.error || 'Erreur ' + response.status);
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
				reject(new Error((xhr.response && xhr.response.error) || 'Erreur ' + xhr.status));
			}
		};
		xhr.onerror = () =>
		{
			setOffline(true);
			reject(new Error(OFFLINE_MESSAGE));
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

function plural(count, singular, pluralForm)
{
	return count + ' ' + (count > 1 ? pluralForm : singular);
}

function formatTimecode(frameNumber)
{
	const fps = (target() && target().fps) || 25;
	const seconds = frameNumber / fps;
	const minutes = Math.floor(seconds / 60);
	const rest = seconds - minutes * 60;
	return String(minutes).padStart(2, '0') + ':' + rest.toFixed(2).padStart(5, '0') + '  ·  image ' + (frameNumber + 1) + ' / ' + target().frame_total;
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
	if (state.offline || (error && error.message === OFFLINE_MESSAGE))
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

	el.targetCaption.textContent = target() ? target().name + ' · ' + target().width + '×' + target().height : 'Propulsé par FaceFusion';
	el.renderLabel.textContent = count ? 'Générer · ' + count : 'Générer';
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
	thumb.style.width = activeButton.offsetWidth + 'px';
	thumb.style.transform = 'translateX(' + (activeButton.offsetLeft - 2) + 'px)';

	el.frameResult.hidden = state.mode === 'original';
	el.faceLayer.hidden = state.mode !== 'original';
	el.compareHandle.hidden = state.mode !== 'compare';
	updateCompare();

	if (!target())
	{
		el.viewerHint.textContent = '';
	}
	else if (state.armedId)
	{
		el.viewerHint.textContent = 'Touchez une personne pour lui donner ce visage';
	}
	else if (state.mode === 'original')
	{
		el.viewerHint.textContent = people().length ? 'Touchez un visage pour le retrouver dans la liste' : '';
	}
	else if (!assignedCount())
	{
		el.viewerHint.textContent = 'Associez un visage source pour voir l’aperçu';
	}
	else
	{
		el.viewerHint.textContent = state.mode === 'compare' ? 'Faites glisser la poignée pour comparer' : 'Aperçu de cette image';
	}
}

function setMode(mode)
{
	state.mode = mode;
	renderMode();

	if (mode !== 'original' && state.previewDirty)
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
		const response = await withDownloadWatch(() => fetch('/api/preview?frame_number=' + state.frameNumber + '&max_size=1920')).catch(() =>
		{
			setOffline(true);
			throw new Error(OFFLINE_MESSAGE);
		});

		if (!response.ok)
		{
			const data = await response.json().catch(() => ({}));
			throw new Error(data.error || 'Aperçu indisponible');
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
		el.frameResult.src = state.previewUrl;

		if (response.headers.get('X-Content-Flagged') === '1')
		{
			toast('Cette image a été floutée par l’analyseur de contenu de FaceFusion.', 'info', 5000);
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
		node.title = person ? 'Personne ' + (index + 1) : 'Visage non répertorié — touchez pour l’ajouter';
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
		el.peopleSubtitle.textContent = 'Ajoutez une cible pour commencer.';
	}
	else if (!list.length)
	{
		el.peopleSubtitle.textContent = isVideo() ? 'Lancez l’analyse de la vidéo.' : 'Aucun visage détecté sur cette image.';
	}
	else
	{
		el.peopleSubtitle.textContent = count + ' sur ' + list.length + ' à remplacer · les autres restent intacts';
	}

	list.forEach((person, index) =>
	{
		const node = document.createElement('li');
		const isNew = !state.renderedPersonIds.has(person.id);
		const appearance = isVideo() ? 'vue ' + person.count + '×' : 'sur la photo';

		renderedIds.add(person.id);
		node.className = 'person' + (person.source_id ? ' is-assigned' : '') + (person.id === state.hoverPersonId ? ' is-hover' : '');
		node.dataset.personId = person.id;

		if (!isNew)
		{
			node.style.animation = 'none';
		}
		node.innerHTML =
			'<button class="person-target" title="' + (isVideo() ? 'Aller à cette image' : 'Personne ' + (index + 1)) + '" style="background-image:url(' + person.thumbnail + ')"><span class="person-number">' + (index + 1) + '</span></button>' +
			ICONS.chevron +
			'<button class="slot' + (person.source_id ? ' is-filled' : '') + '" title="' + (person.source_id ? 'Changer le visage source' : 'Choisir un visage source') + '"' + (person.source_thumbnail ? ' style="background-image:url(' + person.source_thumbnail + ')"' : '') + '>' +
				(person.source_id ? '<span class="mini-remove" title="Retirer">' + ICONS.close + '</span>' : ICONS.plus) +
			'</button>' +
			'<div class="person-meta"><strong>Personne ' + (index + 1) + '</strong><span>' + appearance + ' · ' + (person.source_id ? 'sera remplacée' : 'conservée') + '</span></div>' +
			'<button class="icon-button person-remove" title="Retirer de la liste">' + ICONS.close + '</button>';

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
		node.addEventListener('mouseenter', () => setHoverPerson(person.id));
		node.addEventListener('mouseleave', () => setHoverPerson(null));
		fragment.appendChild(node);
	});

	state.renderedPersonIds = renderedIds;
	el.peopleList.replaceChildren(fragment);
	el.peopleEmpty.hidden = list.length > 0;
	el.peopleFooter.hidden = list.length === 0;
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
		toast('Déposez une photo de visage (jpg, png, webp…)', 'error');
		return;
	}
	try
	{
		const session = await upload('/api/library', images.slice(0, 1), 'files');

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
		toast(session.created ? plural(session.created, 'nouvelle personne ajoutée', 'nouvelles personnes ajoutées') : 'Aucune nouvelle personne sur cette image', session.created ? 'success' : 'info');
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
	const items = library();
	const fragment = document.createDocumentFragment();

	if (state.armedId && !items.some(item => item.id === state.armedId))
	{
		setArmed(null);
	}
	if (!items.length)
	{
		const empty = document.createElement('div');
		empty.className = 'library-empty';
		empty.textContent = 'Déposez ici les photos des nouveaux visages';
		fragment.appendChild(empty);
	}
	items.forEach(item =>
	{
		const node = document.createElement('button');

		node.className = 'library-item' + (item.id === state.armedId ? ' is-armed' : '');
		node.draggable = true;
		node.title = item.name + ' — glissez sur une personne, ou touchez puis choisissez une personne';
		node.innerHTML = '<img alt="" src="' + item.thumbnail + '"><span class="item-name">' + escapeHtml(item.name.replace(/\.[^.]+$/, '')) + '</span><span class="mini-remove remove" title="Supprimer">' + ICONS.close + '</span>';
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
	el.librarySt.querySelectorAll('.library-item').forEach((node, index) => node.classList.toggle('is-armed', library()[index] && library()[index].id === sourceId));
	renderMode();
}

async function addSources(files)
{
	const images = Array.from(files).filter(file => file.type.startsWith('image/'));

	if (!images.length)
	{
		return;
	}
	try
	{
		toast('Analyse de ' + plural(images.length, 'photo', 'photos') + '…', 'info', 1800);
		const session = await upload('/api/library', images, 'files');

		reportRejected(session.rejected);
		setSession(session);

		if (session.added.length)
		{
			toast(plural(session.added.length, 'visage ajouté', 'visages ajoutés') + ' à la bibliothèque');
		}
	}
	catch (error)
	{
		handleError(error);
	}
}

function reportRejected(rejected)
{
	if (rejected && rejected.length)
	{
		toast('Aucun visage trouvé dans : ' + rejected.join(', '), 'error', 5200);
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

	window.addEventListener('dragenter', event =>
	{
		if (dragKind(event) === 'files')
		{
			depth++;
			el.dropOverlayText.textContent = target() ? 'Déposez sur la vidéo/photo pour la remplacer, ou sur une personne' : 'Déposer pour ouvrir';
			el.dropOverlay.hidden = Boolean(target());
			document.body.classList.add('is-dragging');
		}
	});
	window.addEventListener('dragleave', () =>
	{
		depth = Math.max(0, depth - 1);

		if (!depth)
		{
			el.dropOverlay.hidden = true;
			document.body.classList.remove('is-dragging');
		}
	});
	window.addEventListener('dragover', event => event.preventDefault());
	window.addEventListener('drop', event =>
	{
		event.preventDefault();
		depth = 0;
		el.dropOverlay.hidden = true;
		document.body.classList.remove('is-dragging');

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
		toast('Choisissez une photo ou une vidéo', 'error');
		return;
	}
	state.importing = true;
	showHud({ stage: 'Import', message: file.name, progress: 0 }, false);
	try
	{
		const session = await upload('/api/target', [ file ], 'file', progress =>
		{
			showHud(progress < 1 ? { stage: 'Import', message: file.name, progress: progress * 0.8 } : { stage: 'Détection des visages', message: 'Un instant…', progress: 0.9 }, false);
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
			toast(people().length ? plural(people().length, 'visage détecté', 'visages détectés') : 'Aucun visage détecté', people().length ? 'success' : 'info');
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

/* ------------------------------------------------------------ tasks */

function renderTask()
{
	const task = state.session && state.session.task;
	const running = isBusy();

	if (running && task.kind === 'scan')
	{
		el.scanProgress.hidden = false;
		el.scanBar.style.width = Math.round(task.progress * 100) + '%';
		el.scanText.textContent = (task.stage || 'Analyse') + ' · ' + Math.round(task.progress * 100) + '%' + (task.message ? ' · ' + task.message : '');
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
	el.hudStage.textContent = download ? 'Téléchargement du modèle' : task.stage || 'Préparation';
	el.hudMessage.textContent = download ? download.name + ' · ' + formatBytes(download.received) + ' / ' + formatBytes(download.total) : task.message || (indeterminate ? 'Vérification des modèles…' : '');
	el.hudPercent.textContent = indeterminate ? '' : Math.round(progress * 100) + '%';
	el.hudRing.style.strokeDashoffset = String(326.73 * (1 - (indeterminate ? 0.25 : progress)));
	el.hudRingBox.classList.toggle('is-indeterminate', indeterminate);
	el.hudCancel.hidden = !cancellable;
}

function formatBytes(bytes)
{
	if (bytes >= 1024 * 1024 * 1024)
	{
		return (bytes / (1024 * 1024 * 1024)).toFixed(1).replace('.', ',') + ' Go';
	}
	return Math.round(bytes / (1024 * 1024)) + ' Mo';
}

function renderDownload(download)
{
	el.downloadCard.hidden = !download;

	if (download)
	{
		const percent = Math.round(download.progress * 100);

		el.downloadName.textContent = download.name;
		el.downloadPercent.textContent = percent + ' %';
		el.downloadBar.style.width = percent + '%';
		el.downloadSize.textContent = formatBytes(download.received) + ' sur ' + formatBytes(download.total);
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
		toast(task.message, 'error', 6500);
	}
	if (task.status === 'done' && task.kind === 'setup')
	{
		state.options = null;
		request('GET', '/api/options').then(options => { state.options = options; }).catch(() => {});
	}
	if (task.status === 'cancelled')
	{
		toast('Opération annulée', 'info');
	}
	if (task.status === 'done' && task.kind === 'scan')
	{
		toast(people().length ? plural(people().length, 'personne trouvée', 'personnes trouvées') + ' dans la vidéo' : 'Aucun visage trouvé dans la vidéo', people().length ? 'success' : 'info');
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
		media.alt = 'Résultat';
	}
	el.resultName.textContent = output.name;
	el.revealLabel.textContent = state.options && state.options.platform === 'darwin' ? 'Afficher dans le Finder' : 'Afficher dans le dossier';
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
	cpu: 'Processeur (CPU)',
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
			title: 'Remplacement',
			rows:
			[
				{ type: 'select', key: 'face_swapper_model', label: 'Modèle', choices: options.face_swapper_models.map(model => ({ value: model.value, label: model.label + (model.downloaded ? '' : ' — à télécharger') })) },
				{ type: 'select', key: 'face_swapper_pixel_boost', label: 'Résolution du visage', choices: pixelBoosts.map(value => ({ value, label: value.split('x')[0] + ' px' })) },
				{ type: 'slider', key: 'face_swapper_weight', label: 'Force du visage source', min: 0, max: 1, step: 0.05, format: value => Math.round(value * 100) + ' %' }
			],
			footnote: '« À télécharger » : le modèle n’est pas encore sur votre ordinateur, il se téléchargera automatiquement au premier aperçu. Une résolution plus haute est plus nette mais plus lente.'
		},
		{
			title: 'Amélioration',
			rows:
			[
				{ type: 'switch', key: 'face_enhancer', label: 'Rendre les visages plus nets' },
				{ type: 'select', key: 'face_enhancer_model', label: 'Modèle', hidden: !settings.face_enhancer, choices: options.face_enhancer_models.map(model => ({ value: model.value, label: model.label + (model.downloaded ? '' : ' — à télécharger') })) },
				{ type: 'slider', key: 'face_enhancer_blend', label: 'Intensité', hidden: !settings.face_enhancer, min: 0, max: 100, step: 1, format: value => value + ' %' }
			]
		},
		{
			title: 'Masque',
			rows:
			[
				{ type: 'switch', key: 'occlusion', label: 'Respecter mains, cheveux, lunettes' },
				{ type: 'slider', key: 'face_mask_blur', label: 'Fondu des bords', min: 0, max: 1, step: 0.05, format: value => Math.round(value * 100) + ' %' }
			]
		},
		{
			title: 'Reconnaissance',
			rows:
			[
				{ type: 'slider', key: 'reference_face_distance', label: 'Tolérance', min: 0.05, max: 0.95, step: 0.05, format: value => Math.round(value * 100) + ' %' },
				{ type: 'select', key: 'face_detector_model', label: 'Détecteur', choices: options.face_detector_models.map(value => ({ value, label: value === 'many' ? 'Tous (le plus précis)' : value })) },
				{ type: 'slider', key: 'face_detector_score', label: 'Seuil de détection', min: 0.05, max: 0.95, step: 0.05, format: value => Math.round(value * 100) + ' %' }
			],
			footnote: 'Tolérance plus haute : une personne de profil est mieux reconnue, mais deux personnes proches risquent d’être confondues. Seuil plus bas : détecte les petits visages.'
		},
		{
			title: 'Export',
			rows:
			[
				{ type: 'slider', key: 'output_image_quality', label: 'Qualité photo', min: 10, max: 100, step: 1, format: value => value + ' %' },
				{ type: 'slider', key: 'output_video_quality', label: 'Qualité vidéo', min: 10, max: 100, step: 1, format: value => value + ' %' }
			]
		},
		{
			title: 'Performance',
			rows:
			[
				{ type: 'select', key: 'execution_provider', label: 'Calcul', choices: options.execution_providers.map(value => ({ value, label: PROVIDER_LABELS[value] || value })) }
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
	fragment.appendChild(buildThemeGroup());
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

function buildThemeGroup()
{
	const wrapper = document.createElement('div');
	const current = readTheme();

	wrapper.innerHTML =
		'<div class="group-title">Apparence</div>' +
		'<div class="group"><div class="row"><label>Thème</label>' +
		'<span class="select"><select id="setting-theme">' +
		'<option value="auto">Automatique</option><option value="light">Clair</option><option value="dark">Sombre</option>' +
		'</select></span></div></div>';
	const select = wrapper.querySelector('select');

	select.value = current;
	select.addEventListener('change', () =>
	{
		writeTheme(select.value);
		applyTheme();
		requestAnimationFrame(renderMode);
	});
	return wrapper;
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
				// the save may have just downloaded a model, so its « à télécharger » mark must go
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
		// models get downloaded by previews and renders, so the « à télécharger » marks must be fresh
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
	el.targetInput.addEventListener('change', () => el.targetInput.files.length && openTarget(el.targetInput.files[0]));
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
		if (!library().length)
		{
			toast('Ajoutez d’abord des visages sources', 'info');
			return;
		}
		try
		{
			setSession(await request('POST', '/api/people/assign-in-order'));
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
	window.addEventListener('resize', renderMode);
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
		toast('Reconnecté au serveur', 'success');
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
