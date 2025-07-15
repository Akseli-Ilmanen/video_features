
## Installation

### clone the repo and change the working directory
git clone https://github.com/Akseli-Ilmanen/video_features.git
cd video_features

### install environment
conda env create -f environment.yml

### load the environment
conda activate video_features

### extract r(2+1)d features for the sample videos
python main.py \
    feature_type=r21d \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"



### Fixing stuff

If you delete and re-clone this repository, make sure to update the following line in utils/utils.py (function which_ffmpeg):

Change this:
``ffmpeg_path = result.stdout.decode('utf-8').replace('\r\n', '')``

To this:
``ffmpeg_path = result.stdout.decode('utf-8').splitlines()[0]``

in `utils.utils.py` - `where_ffmpeg`
