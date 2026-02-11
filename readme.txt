M2: Model Packaging & Containerization
====================================
Objective: Package the trained model into a reproducible, containerized service.

Status:
- Inference service: DONE (FastAPI) in `src/api.py`
- Requirements: DONE (pinned) in `requirements.txt`
- Dockerfile: DONE in `Dockerfile`

Quick Setup (Local, no Docker)
------------------------------
1) Install deps (conda env example):
   conda run -n rag310 python -m pip install -r requirements.txt

2) Ensure model exists (from DVC):
   dvc pull artifacts.dvc

3) Run API:
   conda run -n rag310 uvicorn src.api:app --host 0.0.0.0 --port 8000

4) Test health:
   curl http://localhost:8000/health

5) Test prediction (multipart image upload):
   curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/image.jpg"

Docker Build/Run
----------------
1) Build image:
   docker build -t cats-dogs-api .

2) Run container:
   docker run -p 8000:8000 cats-dogs-api

3) Verify:
   curl http://localhost:8000/health


M3: CI Pipeline for Build, Test & Image Creation
================================================
Objective: Implement CI to test, build, and publish container images.

Remaining Steps (NOT done yet):
1) Automated Testing
   - Add pytest tests:
     - one data pre-processing function test
     - one model utility/inference function test

2) CI Setup (GitHub Actions / GitLab CI / Jenkins / Tekton)
   - On push/PR: checkout -> install deps -> run pytest -> build Docker image

3) Artifact Publishing
   - Push Docker image to registry (Docker Hub / GHCR / local registry)

Suggested Files to Add:
- `tests/test_preprocess.py`
- `tests/test_inference.py`
- `.github/workflows/ci.yml` (if using GitHub Actions)


M4: CD Pipeline & Deployment
============================
Objective: Deploy the containerized model and automate updates.

Remaining Steps (NOT done yet):
1) Choose deployment target:
   - Docker Compose OR Kubernetes (kind/minikube)

2) Provide manifests:
   - If Docker Compose: `docker-compose.yml`
   - If Kubernetes: `k8s/deployment.yaml` and `k8s/service.yaml`

3) CD / GitOps flow:
   - On main branch merge, pull new image and redeploy

4) Smoke Tests:
   - Post-deploy health check + one prediction call
   - Fail pipeline if tests fail


Notes
-----
- API endpoints implemented:
  - GET /health
  - POST /predict
- Model used: `artifacts/models/baseline_cnn.pt`
- If you change the model path, set environment variable `MODEL_PATH`.
