# 6dof Pose Detection
## Prerequisites
### Segmenter
Git: https://github.com/rstrudel/segmenter
Inside Segmenter directory:
```
pip install .
# python setup.py install
```
```
pip install torch==2.0.1 torchvision==0.15.2 mmcv==1.3.8 timm==0.4.12 numpy==1.23.1
```

### Open3d
#### Headless rendering
https://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html

```
cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF -DPYTHON_EXECUTABLE=$(which python) -DUSE_SYSTEM_ASSIMP=ON ..
make -j$(nproc)
```
https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin

Problem:
```
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $HOME/miniconda3/envs/<env-name>/lib/libstdc++.so.6
```


