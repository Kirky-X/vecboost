# VecBoost Kubernetes 部署配置

## 快速开始

### 1. 构建 Docker 镜像

```bash
docker build -t vecboost:latest .
```

### 2. 应用配置

```bash
kubectl apply -f deployments/kubernetes/
```

### 3. 检查部署状态

```bash
kubectl get pods -l app=vecboost
kubectl logs -l app=vecboost --tail=100
```

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `configmap.yaml` | 应用配置（环境变量） |
| `deployment.yaml` | 无状态部署（CPU 模式） |
| `gpu-deployment.yaml` | GPU 节点部署 |
| `hpa.yaml` | 水平自动扩缩容 |
| `model-cache.yaml` | 模型缓存 StatefulSet |
| `service.yaml` | ClusterIP 服务 |

## 配置参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VECBOOST_HOST` | `0.0.0.0` | 监听地址 |
| `VECBOOST_PORT` | `8080` | 监听端口 |
| `VECBOOST_MODEL_NAME` | `BAAI/bge-m3` | 模型名称 |
| `VECBOOST_DEVICE_TYPE` | `cuda` | 设备类型（cuda/cpu） |
| `VECBOOST_MAX_BATCH_SIZE` | `32` | 最大批处理大小 |
| `VECBOOST_MAX_CONCURRENT_REQUESTS` | `100` | 最大并发请求数 |
| `VECBOOST_TIMEOUT_MS` | `60000` | 超时时间（毫秒） |
| `VECBOOST_LOG_LEVEL` | `info` | 日志级别 |

## GPU 部署

GPU 节点需要安装 NVIDIA 设备插件：

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

## 水平扩展

HPA 配置自动扩缩容策略：
- CPU 利用率 > 70% 时扩容
- 内存利用率 > 80% 时扩容
- 最小 3 个副本，最大 10 个副本

## 监控

### 查看性能指标

```bash
kubectl get pods -l app=vecboost
kubectl top pods -l app=vecboost
```

### 查看健康状态

```bash
curl http://<service-ip>/health
```

## 模型缓存

模型文件存储在 PersistentVolumeClaim `vecboost-models-pvc` 中，容量 20Gi。

## 故障排查

1. **Pod 无法启动**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **GPU 不可用**
   ```bash
   kubectl describe node <node-name>
   kubectl get nodes --selector='gpu-type=nvidia'
   ```

3. **OOM 问题**
   调整 `VECBOOST_GPU_MEMORY_FRACTION` 和资源限制
