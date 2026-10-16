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
		}
	]
};
