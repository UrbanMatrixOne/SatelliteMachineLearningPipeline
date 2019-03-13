# SatelliteMachineLearningPipeline

This documents the tools and steps to build a Satellite Machine Learning Pipeline

## Step 1 : Started a Deep Learning AMI with GPU instance on amazon AWS

## Step 2 Generate Imagery and Labels:

use [label maker](https://github.com/developmentseed/label-maker) to generate imagery:
Full documentation is available [here:](http://devseed.com/label-maker/)
1) install sqlite libraries for tiippecanoe `apt-get install libsqlite3-dev`
2) Install [tippecanoe](https://github.com/mapbox/tippecanoe) from source
3) Install label-maker using `pip install --ignore-installed label_maker`
4) set up config.json
5) `label-maker download`
6) `label-maker labels`
7) `label-maker preview -n 10`

## Step 3:  start jupyter notebook
1) in the server conda env : `conda install nb_conda`
2) from any server conda env : `jupyter notebook --port=8888`
3) from desktop : link local port 8000 to server port 8888 : `ssh -i thisIsmyKey.pem -L 8000:localhost:8888 ubuntu@ec2–34–227–222–100.compute-1.amazonaws.com`
