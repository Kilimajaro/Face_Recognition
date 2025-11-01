## Face Recognition API Project
**[Important]** You can directly run this app by cmds as below: `docker pull linruixu/my-app-repo`, then `docker run linruixu/my-app-repo`.

Or you can build it by yourself as instructed:
The first step, for the installation of the onnx model used in this project,
visit https://drive.usercontent.google.com/download?id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg&export=download&authuser=0 and move this model to the "models" folder (you need to create it by yourself). Then follow the "Basic Docker Workflow" below.

This FastAPI project scaffolds the major pieces of a face recognition pipeline for students. Your job is to implement the missing logic in `util.py` so that the `/face-similarity` endpoint can compare two uploaded face images.

### What You Need to Implement
- `detect_faces(image)` — locate faces in the raw image bytes.
- `compute_face_embedding(face_image)` — generate a numerical embedding per face.
- `detect_face_keypoints(face_image)` — find facial landmarks to support alignment.
- `warp_face(image, homography_matrix)` — warp or align faces with the estimated homography.
- `antispoof_check(face_image)` — score the likelihood that a face is real.
- `calculate_face_similarity(image_a, image_b)` — orchestrate the full pipeline: detect, align, spoof-check, embed, and output a similarity score.

All functions currently can be used except `antispoof_check(face_image)`.

### API Overview
- `GET /health` — simple service liveness check.
- `GET /torch-version` — returns the underlying PyTorch version if installed.
- `POST /face-similarity` — accepts two uploaded images (`image_a`, `image_b`) and responds with a similarity score once the utilities are implemented. Until then, the endpoint returns HTTP 501.

Swagger UI is automatically generated at `http://localhost:5003/docs` (ReDoc at `/redoc`) when the service is running.

### Approaches
- Face detection: Localize each face as a bounding box within the image.
- Face keypoint detection: Extract five canonical landmarks (both eyes, nose tip, and mouth corners).
- Face alignment: Warp every detected face to a normalized 112×112 crop using the RetinaFace five-point template; the aligned 112×112 image is the output passed downstream. Refer to RetinaFace documentation (or ask ChatGPT) for the reference coordinates.
- Face embedding: Transform each aligned face into a 512-dimensional feature vector using the RetinaFace embedding model.
- Similarity scoring: Compare the two embeddings with cosine similarity to obtain the final match score.

[1] https://github.com/deepinsight/insightface/tree/master/model_zoo
[2] https://github.com/serengil/retinaface?tab=readme-ov-file

## Basic Docker Workflow

Ensure Docker Desktop (or another Docker engine) is running before you begin.

- Build the image  
  `docker build -t my-python-app -f Docker/Dockerfile .`
- Run the container with a friendly name  
  `docker run -p 5000:5000 my-python-app`
- Check running containers (optional)  
  `docker ps`
- Tail logs (optional)  
  `docker logs -f fr-container`
- Open the FastAPI Swagger UI  
  `http://localhost:5000/docs`
- Stop and remove the container when finished  
  `docker stop fr-container && docker rm my-python-app`
