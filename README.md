# Installation process
## Python packages
### AV, FFMPEG, and other requirements
```
conda install av -c conda-forge
```

#### aiortc
##### Windows: pray to the Omnissiah and do this
```
pip install --global-option=build_ext --global-option="-ID:\RCSnail\RCSnailPy\3rdparty\include" --global-option="-LD:\RCSnail\RCSnailPy\3rdparty\lib\x64" aiortc
```
##### Linux:
```
sudo apt install libopus-dev --fix-missing
sudo apt install libvpx-dev --fix-missing
pip install aiortc
```

###### Other required packages
```
pip install aiohttp aiohttp-sse-client
pip install pygame Pyrebase opencv-python websockets requests python-firebase firebasedata
pip install ruamel.yaml
```

###### RCSnail library
```
pip install -e ./RCSnailPy
```
