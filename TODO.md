1. Give objects meshes and initial scene without objects to the vlm and let the vlm place all the objects in the scene
2. Give list of objects to the vlm and let in only create object bounding boxes, afterwards place objects inside the bounding boxes
3. Give list of objects + object meshes to the vlm and let it recreate the scene from scratch


TODO:
- Extract visible object from the scenen and upload to huggingface
- use some average metrics on the VLM score