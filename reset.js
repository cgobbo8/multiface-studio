const path = require('path');

module.exports =
{
	run:
	[
		{
			// drop the link to shared models first so the official FaceFusion launcher keeps its files
			when: '{{platform !== "win32"}}',
			method: 'shell.run',
			params:
			{
				message: 'if [ -L .assets ]; then rm .assets; fi'
			}
		},
		{
			when: '{{platform === "win32"}}',
			method: 'shell.run',
			params:
			{
				message: 'if exist .assets rmdir .assets'
			}
		},
		{
			method: 'fs.rm',
			params:
			{
				path: path.resolve(__dirname, '.env')
			}
		}
	]
};
