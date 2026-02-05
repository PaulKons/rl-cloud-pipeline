# RL Cloud Pipeline

## Project Description
This project implements an event-driven cloud architecture for evaluating
Reinforcement Learning (RL) simulations using Kubernetes.

A trained PPO agent controls a PioneerP3DX robot in CoppeliaSim. Jobs are
triggered via Node-RED, processed asynchronously through RabbitMQ, executed
by a Python RL worker, and results are stored in MinIO.

## Architecture Overview
Components:
- Node-RED – workflow orchestration
- RabbitMQ – messaging in amqp (job queue & result notifications)
- RL Worker – Python service running PPO evaluation
- MinIO – object storage for metrics
- Kubernetes (MicroK8s) – container orchestration and persistence

## Job Flow
1. Node-RED publishes a job to RabbitMQ (`rlJobs`)
2. RL Worker consumes the job
3. Worker connects to CoppeliaSim and runs the simulation
4. Metrics are stored in MinIO
5. Worker publishes completion message (`rl.done`)
6. Node-RED receives confirmation

## Technologies Used
- Docker & Docker Compose
- Kubernetes (MicroK8s)
- RabbitMQ
- MinIO
- Node-RED
- Python (stable-baselines3, PPO)
- CoppeliaSim (robot simulation app)

## Repository Structure
rl_docker/
├── docker-compose.yaml
├── rabbitmq-definitions.json
├── rabbitmq.conf
├── rl_worker/
│ ├── Dockerfile
│ ├── worker.py
│ ├── evaluate.py
│ ├── rl_env.py
│ ├── requirements.txt
│ └── ppo_pioneer_multiinput2.zip


## Notes
- CoppeliaSim runs on the host machine
- Kubernetes services communicate internally via ClusterIP
- NodePort is used for UI access (Node-RED, RabbitMQ, MinIO)
- Persistent volumes ensure data survives VM restarts

## Author
Pavlos Konstantinidis
## StudentID
ais25124
