#!/bin/bash
set -e

source ~/.wildfire_aws
export DOCKER_USER="jphyun2019"

echo "Building dev images for ELASTIC_IP=${ELASTIC_IP}"

docker build --platform linux/amd64 \
  -f crew-dojo/Nakama/Dockerfile \
  -t ${DOCKER_USER}/wildfire-nakama:dev \
  crew-dojo/Nakama

docker build --platform linux/amd64 \
  -f wildfire-human-interface/Dockerfile.frontend \
  --build-arg NEXT_PUBLIC_API_URL=http://${ELASTIC_IP}:8080 \
  -t ${DOCKER_USER}/wildfire-frontend:dev \
  ./wildfire-human-interface

docker build --platform linux/amd64 \
  -f wildfire-human-interface/Dockerfile.backend \
  -t ${DOCKER_USER}/wildfire-backend:dev \
  ./wildfire-human-interface

docker build --platform linux/amd64 \
  -f Dockerfile.algorithm \
  -t ${DOCKER_USER}/wildfire-algorithm:dev \
  .

docker push ${DOCKER_USER}/wildfire-frontend:dev
docker push ${DOCKER_USER}/wildfire-backend:dev
docker push ${DOCKER_USER}/wildfire-algorithm:dev
docker push ${DOCKER_USER}/wildfire-nakama:dev

echo "Done. Dev images pushed with :dev tag."
echo "Dev frontend will hit: http://${ELASTIC_IP}:8080"
