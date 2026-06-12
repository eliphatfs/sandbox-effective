Introduction
============
Each line in filelist.txt contains a url to a glb file, for example:
https://github.com/KhronosGroup/glTF-Sample-Assets/raw/refs/heads/main/Models/BoomBox/glTF-Binary/BoomBox.glb

Each GLB contains a mesh that has Y as the up axis.

Your Goal
=========
In the sandbox repository, create a Github Actions CI workflow:

1. Convert each file from Y-up to Z-up.
2. Runs CoACD (https://github.com/SarahWeii/CoACD) on each model.
3. Uploads the CoACD results as GitHub Actions artifacts.
