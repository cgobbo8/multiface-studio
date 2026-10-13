module.exports =
{
	version: '7.0',
	title: 'MultiFace Studio',
	description: 'FaceFusion multi-visages : un nouveau visage pour chaque personne d’une photo ou d’une vidéo, en une seule passe.',
	icon: 'icon.png',
	menu: async (kernel, info) =>
	{
		const installed = info.exists('.env') && info.exists('app/engine');
		const running =
		{
			install: info.running('install.js'),
			start: info.running('start.js'),
			update: info.running('update.js'),
			reset: info.running('reset.js')
		};

		if (running.install)
		{
			return [
			{
				default: true,
				icon: 'fa-solid fa-plug',
				text: 'Installation',
				href: 'install.js'
			}];
		}
		if (installed)
		{
			if (running.start)
			{
				const local = info.local('start.js');

				if (local && local.url)
				{
					return [
					{
						default: true,
						icon: 'fa-solid fa-rocket',
						text: 'Ouvrir le studio',
						href: local.url
					},
					{
						icon: 'fa-solid fa-terminal',
						text: 'Terminal',
						href: 'start.js'
					}];
				}
				return [
				{
					default: true,
					icon: 'fa-solid fa-terminal',
					text: 'Terminal',
					href: 'start.js'
				}];
			}
			if (running.update)
			{
				return [
				{
					default: true,
					icon: 'fa-solid fa-terminal',
					text: 'Mise à jour',
					href: 'update.js'
				}];
			}
			if (running.reset)
			{
				return [
				{
					default: true,
					icon: 'fa-solid fa-terminal',
					text: 'Réinitialisation',
					href: 'reset.js'
				}];
			}
			return [
			{
				default: true,
				icon: 'fa-solid fa-power-off',
				text: 'Démarrer',
				href: 'start.js'
			},
			{
				icon: 'fa-solid fa-rotate',
				text: 'Mettre à jour',
				href: 'update.js'
			},
			{
				icon: 'fa-solid fa-plug',
				text: 'Installer',
				href: 'install.js'
			},
			{
				icon: 'fa-regular fa-circle-xmark',
				text: 'Réinitialiser',
				href: 'reset.js'
			}];
		}
		return [
		{
			default: true,
			icon: 'fa-solid fa-plug',
			text: 'Installer',
			href: 'install.js'
		}];
	}
};
