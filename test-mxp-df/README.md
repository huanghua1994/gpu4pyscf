# Build and test gpu4pyscf MxP DF

## Build Docker image

```bash
git clone https://github.com/huanghua1994/gpu4pyscf.git
cd gpu4pyscf && git checkout mxp-df
cd test-mxp-df/Docker

# Note: 
# 1. if your system has < 16 cores, use a smaller number (do not exceed the number
#    of CPU cores) after -j in build-gpu4pyscf.sh line 26
# 2. Building the docker image takes 50 minutes on an AMD EPYC 7313P 16-core CPU,
#    since the compiling of libxc is very slow
bash ./docker-build.sh
```

## Run tests

Run `docker run --gpus=all -it --shm-size=8g ${username}:ubuntu24.04-gcc13.1-cuda12.8-PySCF-GPU-MxP-DF`, replace `${username}` with your current Linux username. After entering the docker image, you should be in directory `$HOME/gpu4pyscf/test-mxp-df`.

Run `./run-all-tests.sh` to run all tests. Estimated running time:

* 1x H100: 15 minutes
* 1x L40: 1 hour 50 minutes
* 1x RTX 4090: 1 hour 15 minutes
* 1x RTX A5000: 50 minutes
