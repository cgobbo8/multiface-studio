module.exports =
{
	run:
	[
		{
			method: 'shell.run',
			params:
			{
				message: 'git pull'
			}
		},
		{
			// the engine stays on FaceFusion 3.5.4, only the multi-face patch is refreshed from the launcher
			method: 'shell.run',
			params:
			{
				path: 'app/engine',
				message:
				[
					'git checkout --quiet -- .',
					'git clean -fd --quiet',
					'git apply ../patches/facefusion-3.5.4-multi-face.patch'
				]
			}
		}
	]
};
