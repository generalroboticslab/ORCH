#!/bin/bash
set -e

export DOCKER_USER="jzr90"

echo "Building release images..."

docker build --platform linux/amd64 \
  -f crew-dojo/Nakama/Dockerfile \
  -t ${DOCKER_USER}/wildfire-nakama:latest \
  crew-dojo/Nakama

docker build --platform linux/amd64 \
  -f wildfire-human-interface/Dockerfile.frontend \
  --build-arg NEXT_PUBLIC_API_URL=https://crew-wildfire.com/api \
  -t ${DOCKER_USER}/wildfire-frontend:latest \
  ./wildfire-human-interface

docker build --platform linux/amd64 \
  -f wildfire-human-interface/Dockerfile.backend \
  -t ${DOCKER_USER}/wildfire-backend:latest \
  ./wildfire-human-interface

docker build --platform linux/amd64 \
  -f Dockerfile.algorithm \
  -t ${DOCKER_USER}/wildfire-algorithm:latest \
  .

docker push ${DOCKER_USER}/wildfire-nakama:latest
docker push ${DOCKER_USER}/wildfire-frontend:latest
docker push ${DOCKER_USER}/wildfire-backend:latest
docker push ${DOCKER_USER}/wildfire-algorithm:latest

# Capture digests and pin docker-compose.ec2.yml so EC2 always pulls
# exact known-good images instead of whatever :latest resolves to.
echo ""
echo "Pinning docker-compose.ec2.yml to pushed digests..."

export NAKAMA_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${DOCKER_USER}/wildfire-nakama:latest)
export FRONTEND_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${DOCKER_USER}/wildfire-frontend:latest)
export BACKEND_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${DOCKER_USER}/wildfire-backend:latest)
export ALGORITHM_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${DOCKER_USER}/wildfire-algorithm:latest)

python3 << 'EOF'
import re, os

with open('docker-compose.ec2.yml', 'r') as f:
    content = f.read()

for name, digest in [
    ('wildfire-nakama',    os.environ['NAKAMA_DIGEST']),
    ('wildfire-frontend',  os.environ['FRONTEND_DIGEST']),
    ('wildfire-backend',   os.environ['BACKEND_DIGEST']),
    ('wildfire-algorithm', os.environ['ALGORITHM_DIGEST']),
]:
    content = re.sub(rf'image: [^\s]*{name}[^\n]*', f'image: {digest}', content)

with open('docker-compose.ec2.yml', 'w') as f:
    f.write(content)
EOF

echo ""
echo "Done. Images pushed and docker-compose.ec2.yml pinned to:"
echo "  nakama:    ${NAKAMA_DIGEST}"
echo "  frontend:  ${FRONTEND_DIGEST}"
echo "  backend:   ${BACKEND_DIGEST}"
echo "  algorithm: ${ALGORITHM_DIGEST}"
echo ""
echo "Next: scp docker-compose.ec2.yml to EC2, then run:"
echo "  docker compose -f docker-compose.ec2.yml pull && docker compose -f docker-compose.ec2.yml up -d"
