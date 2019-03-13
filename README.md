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

## Step 3:  Use tf Object-Detection api 
#prepare label maker (object detection) 
#### copy onto gpu machine
`aws s3 sync s3://umo-bucket/zambia /storage/zambia`

install obj detection api (on gpu machine)
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

``` 
pip install tensorflow-gpu
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib 
```

#### pycoco 
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

#### compile the probufs 
##### From tensorflow/models/research/
`protoc object_detection/protos/*.proto --python_out=.`

##### From tensorflow/models/research/
`export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

##### test that everything worked
`python object_detection/builders/model_builder_test.py`


##### Create TFRecords for model training
```
cd tensorflow/models/research/object_detection/
cp ~/label-maker/examples/utils/tf_records_generation.py ./
```
##### copy labels.npz to object_detection
`cp /storage/zambia/object_detection/labels.npz ./`
##### Copy your   tiles folders from data to the TOD directory.
`cp -rf /storage/zambia/object_detection/tiles ./`


#### generate tf records
```
python tf_records_generation.py --label_input=labels.npz \
             --train_rd_path=data/train_buildings.record \
             --test_rd_path=data/test_buildings.record
```
##### download inception pretrained model and move ssd_inception... top TOD directory
```
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xcf ssd_inception_v2_coco_2017_11_17.tar.gz
```
##### Create a new folder training in the TOD directory.
`mkdir training`


##### Copy a model configuration file to the training directory. If you aren't using ssd_inception_v2_coco, you'll need to update the configuration file to match your selected model.
`cd training `
(don't use wget)https://github.com/developmentseed/label-maker/blob/master/examples/utils/ssd_inception_v2_coco.config

##### Copy a class definitions file to the data directory.
`cd ../data`
(don't use wget)https://github.com/developmentseed/label-maker/blob/master/examples/utils/building_od.pbtxt

#### now train ! 
```
python model_main.py --alsologtostderr \
              --pipeline_config_path=training/ssd_inception_v2_coco.config \ 
              --model_dir=training/
```              
              


### handy subtasks: 
##### start jupyter:
1) in the server conda env : `conda install nb_conda`
2) from any server conda env : `jupyter notebook --port=8888`
3) from desktop : link local port 8000 to server port 8888 : `ssh -i thisIsmyKey.pem -L 8000:localhost:8888 ubuntu@ec2–34–227–222–100.compute-1.amazonaws.com`

##### add a EB volume to aws EC2 instance

format drive if necessary `sudo mkfs -t ext4 /dev/xvdf `
```
sudo mkdir /newvolume
sudo mount /dev/xvdf /newvolume/
cd /newvolume
df -h .
sudo chown ubuntu /newvolume/
```
to make instance automatically mount volume, backup `/etc/fstab` 
`sudo cp /etc/fstab /etc/fstab.bak`
then add the following line with text editor (emacs)
`/dev/xvdf       /newvolume   ext4    defaults,nofail        0       0`

`sudo mount -a` to test that `/etc/fstab`  is set up correctly.


