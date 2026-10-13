# MultiFace Studio on CPU: FaceFusion 3.5.4 with the multi-face patch, served on port 7870.
# Models are downloaded on the first start into /app/engine/.assets and results go to /app/workspace:
# mount volumes on both to keep them across restarts.
FROM python:3.12-slim-trixie

# FaceFusion fetches and checks its models with curl

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libgl1 libglib2.0-0t64 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/facefusion/facefusion --branch 3.5.4 --single-branch --depth 1 engine
COPY app/patches patches
RUN git -C engine apply ../patches/facefusion-3.5.4-multi-face.patch
# the installer ignores pip failures, so the imports are checked right after
RUN cd engine \
	&& python install.py --onnxruntime default --skip-conda \
	&& python -c "import cv2, fastapi, multipart, numpy, onnxruntime, uvicorn" \
	&& rm -rf /root/.cache/pip
COPY app/studio studio

EXPOSE 7870
CMD [ "python", "studio/server.py", "--host", "0.0.0.0", "--port", "7870", "--remote" ]
