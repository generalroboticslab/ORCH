# WILDFIRE AWS Deployment

Build Docker images locally, push to Docker Hub, pull on AWS.

---

## Your 6 Containers

| Container | Port | Dockerfile | Entry Point |
|-----------|------|------------|-------------|
| caddy | 80, 443 | Pre-built `caddy:2-alpine` | Reverse proxy + HTTPS |
| frontend | 3000 (internal) | `wildfire-human-interface/Dockerfile.frontend` | `node server.js` |
| backend | 8000 (internal) | `wildfire-human-interface/Dockerfile.backend` | `python scripts/fastapi_backend.py` |
| algorithm | 8001 (internal) | `Dockerfile.algorithm` (root) | `python crew-algorithms/.../algorithm_service.py` |
| nakama | 7350 (internal) | `crew-dojo/Nakama/Dockerfile` | Nakama game server |
| postgres | 5432 (internal) | Pre-built `postgres:12.2-alpine` | - |

---

## How They Connect

```
Internet
    │  HTTPS (443) / HTTP→HTTPS redirect (80)
    ▼
Caddy (reverse proxy)
    ├── crew-wildfire.com/         → frontend:3000
    ├── crew-wildfire.com/api/*    → backend:8000
    └── crew-wildfire.com/nakama/* → nakama:7350
            │
            │ HTTP (internal Docker network only)
            ▼
        Backend (8000)
            │
            │ HTTP (ALGORITHM_SERVICE_URL)
            ▼
        Algorithm (8001)
            │
            │ File I/O (shared /tmp volume)
            ▼
        Game subprocess ──────► Nakama (7350)
                                    │
                                    ▼
                               Postgres (5432)
```

**Key environment variables:**
- Frontend: `NEXT_PUBLIC_API_URL` → points to backend via Caddy (set at build time as `https://crew-wildfire.com/api`)
- Backend: `ALGORITHM_SERVICE_URL` → points to algorithm container (internal Docker DNS)
- Algorithm: `OPENAI_API_KEY` → your OpenAI key
- Nakama: `database.address` → points to postgres

**Shared volumes:**
- `./data/tmp` → temp files for backend ↔ algorithm communication (actions, observations, feedback)
- `./data/results` → game results and logs (mounted to `/app/crew-algorithms/crew_algorithms/wildfire_alg/results` in algorithm container)
- `./data/postgres` → Nakama database persistence

---

## Step 1: Install Tools (5 minutes)

```bash
# Install AWS CLI
brew install awscli
aws --version

# Install Docker Desktop (if not installed)
# Download from https://www.docker.com/products/docker-desktop/
docker --version
```

### Configure AWS

1. Go to https://console.aws.amazon.com/iam/
2. Click your username (top right) → "Security credentials"
3. "Create access key" → "Command Line Interface" → Create
4. Copy Access Key ID and Secret Access Key

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, us-east-1, json

aws sts get-caller-identity  # Verify it works
```

### Configure Docker Hub

1. Create account at https://hub.docker.com
2. Login:

```bash
docker login
# Enter your Docker Hub username and password
```

---

## Step 2: Create EC2 Instance (15 minutes)

Run these commands on your Mac:

```bash
# Create SSH key
aws ec2 create-key-pair \
  --key-name wildfire-key \
  --query 'KeyMaterial' \
  --output text > ~/wildfire-key.pem

chmod 400 ~/wildfire-key.pem

# Create security group
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --group-name wildfire-sg \
  --description "WILDFIRE server" \
  --query 'GroupId' \
  --output text)

# Allow ports
# NOTE: Only 22, 80, 443 need to be public.
# All app traffic (frontend, backend, nakama) routes through Caddy on 443.
# Ports 3000, 8000, 8001, 7349-7351 are internal Docker-only.
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr ${MY_IP}/32
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0

# Find Deep Learning AMI (Ubuntu 22.04 with NVIDIA drivers pre-installed)
# This is required for Unity GPU rendering in Docker
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

# Fallback to standard Ubuntu if Deep Learning AMI not found
if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
  echo "Deep Learning AMI not found, using standard Ubuntu (will need manual NVIDIA setup)"
  AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)
fi

echo "AMI: $AMI_ID"

# Launch GPU instance (g4dn.xlarge: 4 vCPU, 16GB RAM, 1 NVIDIA T4 GPU)
# On-demand: ~$0.526/hr | Spot: ~$0.16/hr (70% savings!)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g4dn.xlarge \
  --key-name wildfire-key \
  --security-group-ids $SECURITY_GROUP_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=wildfire-server}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance: $INSTANCE_ID"
echo "Waiting..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Allocate permanent IP
ALLOCATION_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
sleep 10
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOCATION_ID

ELASTIC_IP=$(aws ec2 describe-addresses \
  --allocation-ids $ALLOCATION_ID \
  --query 'Addresses[0].PublicIp' \
  --output text)

echo ""
echo "=========================================="
echo "Server IP: $ELASTIC_IP"
echo "Instance ID: $INSTANCE_ID"
echo "=========================================="

# Save config
cat > ~/.wildfire_aws << EOF
INSTANCE_ID=$INSTANCE_ID
ELASTIC_IP=$ELASTIC_IP
SECURITY_GROUP_ID=$SECURITY_GROUP_ID
ALLOCATION_ID=$ALLOCATION_ID
EOF
```

---

## Step 3: Point Your Domain to the Instance

In your DNS provider (e.g. Cloudflare, Route 53), create two A records pointing to your Elastic IP:

| Name | Type | Value |
|------|------|-------|
| `crew-wildfire.com` | A | `<your Elastic IP>` |
| `www.crew-wildfire.com` | A | `<your Elastic IP>` |

Caddy will automatically obtain and renew HTTPS certificates from Let's Encrypt once DNS is live. This requires port 80 to be publicly accessible (which the security group above allows).

---

## Step 4: Build Unity for Linux

The algorithm container runs Unity game builds. Since Docker runs Linux, you need **Linux builds**.

**IMPORTANT: Do this BEFORE building Docker images** - the builds get baked into the algorithm container.

In Unity Editor on your Mac:
1. Open the Wildfire project (`crew-dojo/Unity/Wildfire`)
2. **File → Build Settings**
3. Switch platform to **Linux**
4. Architecture: **x86_64**
5. **DO NOT check "Server Build"** - we need graphics rendering for visual observations!
6. Build to: `crew-dojo/Builds/Wildfire-StandaloneLinux64-Server/`

**Before building**, update the Nakama config for Docker networking:
1. Open `crew-dojo/Unity/Assets/Examples/Wildfire/Configs/NakamaConfig.asset`
2. Set `Host: 172.23.0.50` (the Nakama container's Docker IP)
3. Save and build

The output should be `Unity.x86_64` executable.

Verify it exists:
```bash
ls -la crew-dojo/Builds/Wildfire-StandaloneLinux64-Server/Unity.x86_64
```

The `.dockerignore` is configured to include this folder in the algorithm image.

---

## Step 5: Build & Push Docker Images (15 minutes)

On your Mac, in the CREW-dev folder:

**IMPORTANT:** If you're on an Apple Silicon Mac (M1/M2/M3), you MUST use `--platform linux/amd64` to build images that work on EC2.

Use the release build script, which builds all images, pushes them to Docker Hub, and automatically pins `docker-compose.ec2.yml` to exact image digests (so a future push to `:latest` on Docker Hub can never silently swap in a different image):

```bash
cd /Users/jonathanhyun/Documents/Lab/CREW-dev
./build-release.sh
```

This script:
1. Builds all 4 images for `linux/amd64`
2. Pushes them to Docker Hub under `:latest`
3. Captures each image's SHA256 digest
4. Updates `docker-compose.ec2.yml` in place with pinned `@sha256:...` references

**Note:** `NEXT_PUBLIC_API_URL` is baked into the frontend image at build time as `https://crew-wildfire.com/api`. If your domain changes, you must rebuild the frontend image.

---

## Step 6: Set Up EC2 with GPU Support (15 minutes)

```bash
# SSH into your server
source ~/.wildfire_aws
ssh -i ~/wildfire-key.pem ubuntu@$ELASTIC_IP
```

Once connected to EC2:

```bash
# Verify GPU is available (Deep Learning AMI has drivers pre-installed)
nvidia-smi

# If nvidia-smi doesn't work, install NVIDIA drivers:
# sudo apt-get update && sudo apt-get install -y nvidia-driver-535

# Install NVIDIA Container Toolkit (for Docker GPU access)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify Docker can access GPU
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install X server (not included in Deep Learning AMI by default)
sudo apt-get install -y xserver-xorg x11-xserver-utils

# Set up X server for Unity GPU rendering
# Find the GPU's PCI bus ID
GPU_BUS_ID=$(nvidia-xconfig --query-gpu-info | grep "PCI BusID" | sed 's/.*PCI BusID : //')
echo "GPU Bus ID: $GPU_BUS_ID"

# Write xorg.conf for headless GPU rendering
sudo tee /etc/X11/xorg.conf << XEOF
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    BusID          "$GPU_BUS_ID"
    Option         "AllowEmptyInitialConfiguration" "True"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Virtual     1280 1024
    EndSubSection
EndSection
XEOF

# Start X on display :0 (run in background)
sudo X :0 -config /etc/X11/xorg.conf &

# Allow local connections to X server
export DISPLAY=:0
xhost +local:

# Create project folder
mkdir -p ~/wildfire
cd ~/wildfire
mkdir -p data/{results,logs,tmp,postgres}

# Create .env file with your OpenAI key
cat > .env << EOF
OPENAI_API_KEY=<your-openai-api-key-here>

EOF

nano .env  # Edit to confirm/update the key
```

---

## Step 7: Copy Config Files to EC2

On your Mac (run from the CREW-dev repo root after `./build-release.sh`):

```bash
source ~/.wildfire_aws

# Copy the compose file (with pinned digests) and Caddyfile
scp -i ~/wildfire-key.pem docker-compose.ec2.yml ubuntu@${ELASTIC_IP}:~/wildfire/docker-compose.yml
scp -i ~/wildfire-key.pem Caddyfile ubuntu@${ELASTIC_IP}:~/wildfire/Caddyfile
```

---

## Step 8: Deploy (5 minutes)

On EC2:

```bash
cd ~/wildfire

# Pull the pinned images
docker compose pull

# Start everything
docker compose up -d

# Watch startup — services come up in dependency order:
# postgres → nakama (healthy) → algorithm (healthy, ~15s) → backend (healthy) → frontend (healthy) → caddy
docker compose ps

# View logs
docker compose logs -f
# Ctrl+C to stop watching
```

Verify services are healthy from EC2:

```bash
# Check all containers are healthy (not just "Up")
docker compose ps

# Test internal health endpoints directly inside containers
docker exec wildfire-backend python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read())"
docker exec wildfire-algorithm python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health').read())"

# Check Caddy is running and its admin API is up
docker exec wildfire-caddy wget -qO- http://localhost:2019/
```

---

## Step 9: Test from Internet

On your Mac:

```bash
# Test HTTPS (Caddy auto-obtained a cert from Let's Encrypt)
curl https://crew-wildfire.com/api/health
curl https://www.crew-wildfire.com/api/health

# Open in browser
open https://crew-wildfire.com
```

If HTTPS isn't working yet, check that DNS has propagated:
```bash
dig crew-wildfire.com +short  # Should return your Elastic IP
```

Share with users:
- Site: `https://crew-wildfire.com`
- Nakama WebSocket: `wss://crew-wildfire.com/nakama/` (proxied through Caddy)

---

## Step 10: Auto-Start on Reboot

On EC2:

```bash
sudo tee /etc/systemd/system/wildfire.service << 'EOF'
[Unit]
Description=WILDFIRE
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/wildfire
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable wildfire.service
```

---

## Accessing Data

From your Mac:

```bash
source ~/.wildfire_aws

# Download results
scp -i ~/wildfire-key.pem -r ubuntu@$ELASTIC_IP:~/wildfire/data/results ./results

# Download logs
scp -i ~/wildfire-key.pem -r ubuntu@$ELASTIC_IP:~/wildfire/data/logs ./logs

# View logs live
ssh -i ~/wildfire-key.pem ubuntu@$ELASTIC_IP "docker logs -f wildfire-algorithm"
```

---

## Releasing Updates

When you change code, use the release script which rebuilds, pushes, and re-pins the compose file, then copy it to EC2:

```bash
cd /Users/jonathanhyun/Documents/Lab/CREW-dev

# Build, push, and update docker-compose.ec2.yml with new pinned digests
./build-release.sh

# Copy updated compose file to EC2 and redeploy
source ~/.wildfire_aws
scp -i ~/wildfire-key.pem docker-compose.ec2.yml ubuntu@${ELASTIC_IP}:~/wildfire/docker-compose.yml
ssh -i ~/wildfire-key.pem ubuntu@${ELASTIC_IP} "cd ~/wildfire && docker compose pull && docker compose up -d"
```

**If you change the Caddyfile**, copy it too:
```bash
scp -i ~/wildfire-key.pem Caddyfile ubuntu@${ELASTIC_IP}:~/wildfire/Caddyfile
ssh -i ~/wildfire-key.pem ubuntu@${ELASTIC_IP} "docker exec wildfire-caddy caddy reload --config /etc/caddy/Caddyfile"
```

**Note:** The frontend's `NEXT_PUBLIC_API_URL` is baked into the image at build time as `https://crew-wildfire.com/api`. It does not need to change when you move to a new instance, as long as the domain stays the same.

---

## Management

On EC2:

```bash
cd ~/wildfire
docker compose ps             # Status + health
docker compose logs -f        # All logs
docker compose logs -f frontend  # Single service logs
docker compose restart        # Restart all
docker compose stop           # Stop all
docker compose start          # Start all
```

---

## Shutdown & Startup (Save Money!)

When not using the server, **stop it to save ~$0.50/hr** (GPU instances are expensive).

### Shutdown (from Mac)

```bash
source ~/.wildfire_aws

# Stop the instance (data is preserved on EBS)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Verify it's stopped
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name' --output text
# Should show: stopped
```

**What's preserved:**
- All data in `~/wildfire/data/` (results, postgres, tmp)
- Docker images (already pulled)
- docker-compose.yml and .env configuration
- System packages and NVIDIA drivers
- Caddy TLS certificates (stored in `caddy_data` Docker volume)

**What's NOT preserved:**
- Running X server process (must restart)
- Active game sessions (users must start new games)

### Startup (from Mac)

```bash
source ~/.wildfire_aws

# Start the instance
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Wait for it to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

echo "Server ready at: https://crew-wildfire.com"
```

### After Startup - Restart X Server (on EC2)

**IMPORTANT:** The X server doesn't auto-start on reboot. You must SSH in and restart it.

```bash
# SSH into server
source ~/.wildfire_aws
ssh -i ~/wildfire-key.pem ubuntu@$ELASTIC_IP

# Check if X is already running
ps aux | grep "X :0"

# If not running, start X server
sudo X :0 -config /etc/X11/xorg.conf &

# Wait a moment, then allow local connections
sleep 2
export DISPLAY=:0
xhost +local:

# Restart algorithm container to pick up X server
cd ~/wildfire
docker compose restart algorithm

# Verify everything is working
docker compose ps
docker exec wildfire-algorithm nvidia-smi  # GPU should show Unity process
```

### Quick Startup Script (Optional)

Create this script on EC2 to make restarts easier:

```bash
cat > ~/start-wildfire.sh << 'EOF'
#!/bin/bash
set -e

# Start X server if not running
if ! pgrep -x "X" > /dev/null; then
    echo "Starting X server..."
    sudo X :0 -config /etc/X11/xorg.conf &
    sleep 3
fi

# Allow X connections
export DISPLAY=:0
xhost +local:

# Restart algorithm container to pick up X server
cd ~/wildfire
docker compose restart algorithm

echo "WILDFIRE ready!"
docker compose ps
EOF

chmod +x ~/start-wildfire.sh
```

Then after each instance startup, just run:
```bash
~/start-wildfire.sh
```

---

## What Happens on Crashes/Restarts

**Container crashes:**
- Docker automatically restarts it (`restart: unless-stopped`)
- In-memory game state is lost (active lobbies, chats)
- Users just start a new game - no manual intervention needed

**Startup ordering:**
- Containers start in dependency order with health checks
- `docker compose up -d` waits for each service to be healthy before starting the next
- Algorithm gets a 120s grace period for CUDA/PyTorch/Unity load time

**Server reboots:**
- systemd service starts docker compose automatically
- Same as above - users start fresh games

**Persistent data (survives restarts):**
- `data/results/` - Game results
- `data/logs/` - Application logs
- `data/tmp/` - Actions, feedback, observations files
- `data/postgres/` - Database
- `caddy_data` volume - TLS certificates (Caddy won't need to re-issue)

**Not persistent (lost on restart):**
- Active game sessions (must start new game)
- In-memory chat history
- Running game processes

This is fine because games are short sessions. If something crashes mid-game, users refresh and start a new lobby.

---

## Dependencies Reference

### Frontend (`wildfire-human-interface/package.json`)
- Next.js 15, React 18
- Radix UI components
- Tailwind CSS

### Backend (`wildfire-human-interface/requirements.txt`)
- FastAPI, uvicorn
- torch, torchrl
- hydra-core, omegaconf
- openai, requests

### Algorithm (`Dockerfile.algorithm` uses)
- `crew-algorithms/requirements-full.txt` - main deps
- `crew-algorithms/crew_algorithms/wildfire_alg/requirements.txt` - wildfire-specific
- Git packages: ml-agents, whisper

### Nakama (`crew-dojo/Nakama/Dockerfile`)
- heroiclabs/nakama:3.15.0 base image
- Custom Go plugins built from `crew-dojo/Nakama/*.go`
- Configuration from `crew-dojo/Nakama/local.yml`

All dependencies are installed inside the Docker images. No manual pip/npm install needed on EC2.

---

## Cost

**GPU Instance (g4dn.xlarge) - Required for Unity rendering:**
- On-demand: ~$0.526/hr (~$379/month if running 24/7)
- **Spot Instance: ~$0.16/hr (~$115/month)** - 70% savings!

**Cost Saving Tips:**

1. **Use Spot Instances** (saves 60-70%):
   ```bash
   # Launch as spot instance instead of on-demand
   INSTANCE_ID=$(aws ec2 run-instances \
     --image-id $AMI_ID \
     --instance-type g4dn.xlarge \
     --key-name wildfire-key \
     --security-group-ids $SECURITY_GROUP_ID \
     --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
     --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=wildfire-server}]' \
     --query 'Instances[0].InstanceId' \
     --output text)
   ```

2. **Stop when not in use**:
   ```bash
   source ~/.wildfire_aws
   aws ec2 stop-instances --instance-ids $INSTANCE_ID  # Saves ~$0.50/hr
   aws ec2 start-instances --instance-ids $INSTANCE_ID  # When ready to use
   ```

3. **Schedule auto-stop** (optional):
   - Use AWS Instance Scheduler or Lambda to stop instances at night

**Spot Instance Notes:**
- Spot instances can be interrupted with 2-minute warning (rare for g4dn)
- Use "persistent" spot type so it auto-restarts after interruption
- Data on EBS volumes is preserved

---

## Troubleshooting

**Site not loading (HTTPS):**
- Check DNS has propagated: `dig crew-wildfire.com +short` should return your Elastic IP
- Check Caddy logs: `docker logs wildfire-caddy --tail=100`
- Verify port 80 and 443 are open in security group (needed for Let's Encrypt)
- Check Caddy admin API: `docker exec wildfire-caddy wget -qO- http://localhost:2019/`

**Service stuck in "health: starting":**
- Algorithm is given 120s to start (CUDA/PyTorch/Unity load time) — wait for it
- Check logs: `docker logs wildfire-algorithm --tail=50`
- If health check keeps failing: `docker exec wildfire-algorithm python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"`

**Backend can't reach algorithm:**
- Check `ALGORITHM_SERVICE_URL=http://algorithm:8001` in docker-compose
- Both must be on same Docker network (`wildfire-net`)

**Temp file communication fails:**
- Both backend and algorithm need `./data/tmp:/tmp` volume mount
- Check files are being created: `ls ~/wildfire/data/tmp/`

**Nakama won't start:**
- Check postgres is running: `docker compose logs postgres`
- Nakama needs postgres to be healthy before it can run migrations
- Check Nakama logs: `docker compose logs nakama`

**Game subprocess can't connect to Nakama:**
- Ensure algorithm container has IP 172.23.0.100 (matches host.docker.internal in Nakama config)
- All containers must be on the same network (wildfire-net)

**Unity not rendering (GPU issues):**
- Check GPU is visible: `docker exec wildfire-algorithm nvidia-smi`
- Check X server is running on host: `ps aux | grep X`
- Check DISPLAY is set: `docker exec wildfire-algorithm echo $DISPLAY`
- Verify docker-compose has `runtime: nvidia` for algorithm service

---

## Migrating to GPU Instance (from non-GPU)

If you have an existing non-GPU instance and need to migrate:

```bash
# On your Mac - save your current config
source ~/.wildfire_aws
OLD_INSTANCE_ID=$INSTANCE_ID
OLD_ELASTIC_IP=$ELASTIC_IP

# Note: You'll reassociate the same Elastic IP with the new instance
# so your URL stays the same

# 1. Create new GPU instance (follow Step 2 with g4dn.xlarge)
# This will create a NEW instance

# 2. Move Elastic IP to new instance
aws ec2 disassociate-address --public-ip $OLD_ELASTIC_IP
aws ec2 associate-address --instance-id $NEW_INSTANCE_ID --public-ip $OLD_ELASTIC_IP

# 3. Update your config
cat > ~/.wildfire_aws << EOF
INSTANCE_ID=$NEW_INSTANCE_ID
ELASTIC_IP=$OLD_ELASTIC_IP
SECURITY_GROUP_ID=$SECURITY_GROUP_ID
ALLOCATION_ID=$ALLOCATION_ID
EOF

# 4. Follow Step 6 to set up the new instance

# 5. Delete old instance when confirmed working
aws ec2 terminate-instances --instance-ids $OLD_INSTANCE_ID
```

---

## Delete Everything (Cleanup)

To completely remove AWS resources and stop all billing:

```bash
source ~/.wildfire_aws

# 1. Terminate EC2 instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# 2. Wait for termination
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

# 3. Release Elastic IP (stops $3.65/month charge for unused IPs)
aws ec2 release-address --allocation-id $ALLOCATION_ID

# 4. Delete security group
aws ec2 delete-security-group --group-id $SECURITY_GROUP_ID

# 5. Delete key pair
aws ec2 delete-key-pair --key-name wildfire-key
rm ~/wildfire-key.pem

# 6. Clean up local config
rm ~/.wildfire_aws

echo "All AWS resources deleted!"
```

**Warning:** This deletes everything including any data stored on the instance.
