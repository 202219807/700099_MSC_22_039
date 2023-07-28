**Install required packages**

Linux:
```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
```

Python dependencies:

`python3 -m pip install --upgrade pip setuptools psutil wheel`

`python3 -m pip install gfootball`

`python3 -m pip install --upgrade pip setuptools wheel`

`python3 -m pip install tensorflow==1.15.*` or `python3 -m pip install tensorflow-gpu==1.15.*`

`python3 -m pip install dm-sonnet==1.* psutil`

`python3 -m pip install git+https://github.com/openai/baselines.git@master`.
