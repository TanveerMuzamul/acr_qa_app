python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
set PYTHONPATH=%cd%
python -m pytest -v --maxfail=50
python run.py

git init
git branch -M main
git add .
git commit -m "Final app update"
git remote remove origin
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main --force

git checkout -b final-cleanup
git add .
git commit -m "Cleanup final application"
git push -u origin final-cleanup
git checkout main
git merge final-cleanup
git push origin main

docker build -t acr-qa-app .
docker run -p 5000:5000 --env-file .env acr-qa-app
docker compose up --build

aws configure
aws s3 mb s3://YOUR_BUCKET_NAME
aws s3 cp .env s3://YOUR_BUCKET_NAME/.env

aws ecr create-repository --repository-name acr-qa-app
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com
docker tag acr-qa-app:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/acr-qa-app:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/acr-qa-app:latest

ssh -i YOUR_KEY.pem ec2-user@YOUR_EC2_PUBLIC_IP
docker pull YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/acr-qa-app:latest
docker run -d -p 80:5000 --name acr-qa-app YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/acr-qa-app:latest
