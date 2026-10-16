const path = require('path');

module.exports =
{
	daemon: true,
	run:
	[
		{
			method: 'shell.run',
			params:
			{
				conda:
				{
					path: path.resolve(__dirname, '.env')
				},
				message: 'python studio/server.py --port {{port}}',
				on:
				[
					{
						event: '/(http:\\/\\/[0-9.:]+)/',
						done: true
					}
				]
			}
		},
		{
			method: 'local.set',
			params:
			{
				url: '{{input.event[1]}}'
			}
		}
	]
};
