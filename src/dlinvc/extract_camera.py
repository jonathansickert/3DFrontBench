import bpy
import json

cam = bpy.context.scene.camera
loc = cam.location
rot = cam.rotation_euler

with open("/Users/jonathansickert/git/DLinVC/assets/camera.json", "w") as file:
    json.dump({
        "loc" : [x for x in loc],
        "rot" : [r for r in rot],
        "focal_length" : cam.data.lens,
    }, file)

