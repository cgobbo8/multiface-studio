const fs = require('fs');
const path = require('path');

// models already downloaded by the official FaceFusion launcher, reused instead of downloading ~3 GB again
const OFFICIAL_ASSETS_PATH = path.resolve(__dirname, '..', 'facefusion-pinokio.git', 'facefusion', '.assets');

function install(kernel)
{
	const { platform, gpu } = kernel;

	if (platform === 'linux' && gpu === 'amd')
	{
		return 'python install.py --onnxruntime migraphx';
	}
	if (platform === 'win32' && gpu === 'amd')
	{
		return 'python install.py --onnxruntime directml';
	}
	if ((platform === 'linux' || platform === 'win32') && gpu === 'intel')
	{
		return 'python install.py --onnxruntime openvino';
	}
	if ((platform === 'linux' || platform === 'win32') && gpu === 'nvidia')
	{
		return 'python install.py --onnxruntime cuda';
	}
	return 'python install.py --onnxruntime default';
}

module.exports = async kernel =>
{
	const config =
	{
		requires:
		{
			bundle: 'ai'
		},
		run:
		[
			{
				when: '{{!exists("app/engine")}}',
				method: 'shell.run',
				params:
				{
					path: 'app',
					message:
					[
						'git clone https://github.com/facefusion/facefusion --branch 3.5.4 --single-branch engine',
						'git -C engine apply ../patches/facefusion-3.5.4-multi-face.patch'
					]
				}
			},
			{
				when: '{{platform !== "win32" && ' + fs.existsSync(OFFICIAL_ASSETS_PATH) + ' && !exists("app/engine/.assets")}}',
				method: 'shell.run',
				params:
				{
					path: 'app/engine',
					message: 'ln -s "' + OFFICIAL_ASSETS_PATH + '" .assets'
				}
			},
			{
				when: '{{platform === "win32" && ' + fs.existsSync(OFFICIAL_ASSETS_PATH) + ' && !exists("app/engine/.assets")}}',
				method: 'shell.run',
				params:
				{
					path: 'app/engine',
					message: 'mklink /J .assets "' + OFFICIAL_ASSETS_PATH + '"'
				}
			},
			{
				method: 'shell.run',
				params:
				{
					message: 'conda install conda=25.5.1 --yes'
				}
			},
			{
				method: 'shell.run',
				params:
				{
					message: 'conda install python=3.12 --yes',
					conda:
					{
						path: path.resolve(__dirname, '.env')
					}
				}
			},
			{
				method: 'shell.run',
				params:
				{
					message: 'conda install conda-forge::ffmpeg=7.0.2 conda-forge::libvorbis=1.3.7 --yes',
					conda:
					{
						path: path.resolve(__dirname, '.env')
					}
				}
			},
			{
				when: '{{(platform === "linux" || platform === "win32") && gpu === "intel"}}',
				method: 'shell.run',
				params:
				{
					message: 'conda install conda-forge::openvino=2025.1.0 --yes',
					conda:
					{
						path: path.resolve(__dirname, '.env')
					}
				}
			},
			{
				when: '{{(platform === "linux" || platform === "win32") && gpu === "nvidia"}}',
				method: 'shell.run',
				params:
				{
					message:
					[
						'conda install nvidia/label/cuda-12.9.1::cuda-runtime nvidia/label/cudnn-9.10.0::cudnn --yes',
						'pip install tensorrt==10.12.0.36 --extra-index-url https://pypi.nvidia.com'
					],
					conda:
					{
						path: path.resolve(__dirname, '.env')
					}
				}
			},
			{
				method: 'shell.run',
				params:
				{
					message: install(kernel),
					path: 'app/engine',
					conda:
					{
						path: path.resolve(__dirname, '.env')
					},
					on:
					[
						{
							event: '/error:/i',
							break: false
						}
					]
				}
			},
			{
				method: 'input',
				params:
				{
					title: 'Installation terminée',
					description: 'Revenez au tableau de bord pour démarrer MultiFace Studio.'
				}
			}
		]
	};

	return config;
};
