# SMART-STORAGE
An integrated approach to modeling and storage, using sequential model representations.

## Status

We are currently porting the code from another library into a more microservice approach.

## Docker Setup

### Docker Installation

All of the instructions are available [here][1]. For Ubuntu, make sure any existing installations are removed with:

```bash
$ sudo apt-get remove docker docker-engine docker.io
```

Then, update and install the latest version of Docker CE.

```bash
$ sudo apt-get update && sudo apt-get install docker-ce
```

To test that your installation is working, run:

```bash
$ sudo docker run hello-world
```

### Docker build

After you have docker up and running and you have cloned this repository, you will need to build the image. The image can be built with:

```bash
sudo docker build -t smartstorage:v0.1 .
```

You can replace the name and tag with an alternative if you wish.

### Using the code in Docker

Once the image has been built, you can use the docker image to work with the code. My preferred approach for development mounts the `smartstorage` folder as follows:

```bash
sudo docker run -it -u $(id -u):$(id -g) --rm --mount type=bind,src=/path/to/smartstore,dst=/workdir/smartstore smartstorage:v0.1 /bin/bash
```

The environment is setup for python3 or ipython.

[1]: https://docs.docker.com/install/
