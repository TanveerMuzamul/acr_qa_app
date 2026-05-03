# MRI ACR QA Pipeline

Cloud-ready Flask application for automated MRI ACR phantom quality assurance. The application uploads ZIP files containing MR DICOM data, validates the dataset, runs ACR checks, and reports only `PASS` or `FAIL` outcomes.

## Run Application Locally
Run
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
pytest
python run.py
http://127.0.0.1:5000


## Build and Run with Docker (Local)
Run
docker build -t mri-qa .
docker run -p 5000:5000 mri-qa
http://127.0.0.1:5000


## Run with Docker Compose
Run
docker compose up --build


## Stop Docker
Run
docker ps
docker stop CONTAINER_ID
docker compose down


## Push Image to Docker Hub
Run
docker login
docker tag mri-qa tanveermuzamul/mri-qa
docker push tanveermuzamul/mri-qa


## Run Application on AWS using Docker (Combined)
Run

ssh -i mri-key.pem ubuntu@16.171.147.192

sudo apt update
sudo apt install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu

exit
ssh -i mri-key.pem ubuntu@16.171.147.192

docker pull tanveermuzamul/mri-qa
docker rm -f mri-qa 2>/dev/null || true
docker run -d -p 80:5000 --name mri-qa tanveermuzamul/mri-qa

http://16.171.147.192


## Check AWS Container
Run
docker ps
docker logs mri-qa


## Stop or Restart on AWS
Run
docker stop mri-qa
docker start mri-qa


## Run Fresh on AWS
Run
docker rm -f mri-qa
docker pull tanveermuzamul/mri-qa
docker run -d -p 80:5000 --name mri-qa tanveermuzamul/mri-qa