docker run --gpus all \
    --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it \
    -v $(pwd):/my-repo \
    -w /my-repo \
    --rm comps