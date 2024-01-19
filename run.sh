docker build . -t comps
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/my-repo --rm comps