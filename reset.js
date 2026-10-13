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
				message: 'if [ -L app/engine/.assets ]; then rm app/engine/.assets; fi'
			}
		},
		{
			when: '{{platform === "win32"}}',
			method: 'shell.run',
			params:
			{
				message: 'if exist app\\engine\\.assets rmdir app\\engine\\.assets'
			}
		},
		{
			method: 'fs.rm',
			params:
			{
				path: 'app/engine'
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
