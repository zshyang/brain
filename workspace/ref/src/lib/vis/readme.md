```
.
|-- develop_x3d : visualize the split mesh
|-- htmlviewer
|-- threejs
|-- mesh_viewer.py : from MeshCNN not very useful
|-- shapenet_viewer.py
```
chrome and three.js have limited the number of canvases to 
16.
The way to work around is as following:
https://stackoverflow.com/questions/41919341/is-there-a-limit-to-the-number-of-three-webglrenderer-instances-in-a-page
And it will take me time to figure out how to add scenes in
a canvas. and create a table inside of a canvas.
