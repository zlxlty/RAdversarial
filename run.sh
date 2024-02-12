docker run \
    --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it \
    -v $(pwd):/my-repo \
    -v /home/comps-shared/datasets/:/dataset \
    -w /my-repo \
    --rm comps