docker run  --gpus all \
    --user $(id -u):$(id -g) \
    --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it \
    -v $(pwd):/my-repo \
    -v /home/comps-shared/datasets/:/dataset \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -w /my-repo \
    --rm comps