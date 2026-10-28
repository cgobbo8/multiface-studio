const path = require('path');

module.exports =
{
	run:
	[
		{
			// the engine and the studio live in this repository, one pull updates both
			method: 'shell.run',
			params:
			{
				message: 'git pull'
			}
		},
		{
			// a new version may bring new packages for the studio
			method: 'shell.run',
			params:
			{
				message: 'pip install -r studio/requirements.txt',
				conda:
				{
					path: path.resolve(__dirname, '.env')
				}
			}
		}
	]
};
