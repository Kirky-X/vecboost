# VecBoost 水平扩展最佳实践

本指南提供 VecBoost 文本向量化服务在 Kubernetes 环境中进行水平扩展的配置建议和最佳实践。

## 1 Horizontal Pod Autoscaler 配置

### 1.1 基础 HPA 配置

HPA（Horizontal Pod Autoscaler）根据资源利用率自动调整 Pod 副本数量。以下是针对不同工作负载场景的推荐配置。

#### 场景一：高并发在线服务

对于需要处理大量并发请求的在线服务，建议采用以下配置：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vecboost-hpa-online
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vecboost
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

此配置的关键参数说明如下。`minReplicas: 5` 确保即使在最低负载时也有足够的副本来处理请求和维护高可用性。`maxReplicas: 20` 设置了上限以控制成本。CPU 利用率阈值设为 60%，这意味着当 CPU 使用率超过 60% 时就会触发扩容，相比默认的 80% 阈值能更早地响应负载增长。Scale-up 策略选择 `Max` 意味着在多个扩容策略中会选择最快的扩容速度。`stabilizationWindowSeconds: 0` 使扩容立即生效，确保服务能快速响应流量峰值。

#### 场景二：批处理任务优先

对于以批处理任务为主、偶尔有负载尖峰的工作负载，建议采用以下配置：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vecboost-hpa-batch
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vecboost
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 85
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Percent
        value: 20
        periodSeconds: 120
      - type: Pods
        value: 1
        periodSeconds: 120
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

批处理场景下提高阈值并增加稳定窗口可以避免不必要的频繁扩缩容。扩容稳定窗口设置为 60 秒，允许负载短暂峰值通过而不立即扩容。缩容稳定窗口延长到 600 秒，防止批处理任务完成后立即缩容。

### 1.2 自定义指标 HPA 配置

对于需要更精细控制的场景，可以配置基于自定义指标的 HPA：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vecboost-hpa-custom
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vecboost
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: vecboost_pending_requests
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
```

自定义指标 `vecboost_pending_requests` 表示待处理的请求队列长度。当平均待处理请求数超过 50 时触发扩容，这种方式比单纯依赖 CPU 利用率更能反映实际业务负载。

## 2 模型缓存策略

### 2.1 共享模型存储架构

在多实例部署中，共享模型文件可以显著减少内存占用和启动时间。推荐的架构如下：

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Pod 1     │    │   Pod 2     │    │   Pod 3     │      │
│  │             │    │             │    │             │      │
│  │  /models    │◄──►│  /models    │◄──►│  /models    │      │
│  │  (bind)     │    │  (bind)     │    │  (bind)     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│          │                │                │                 │
│          └────────────────┼────────────────┘                 │
│                           ▼                                  │
│              ┌─────────────────────────┐                    │
│              │   NFS / CephFS / EFS    │                    │
│              │   Model Storage (20Gi)  │                    │
│              └─────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

使用共享存储的关键优势包括：模型文件只需下载一次，后续 Pod 启动无需重复下载；所有 Pod 共享同一份模型文件，节省存储空间；更新模型时只需更新共享存储中的文件，所有 Pod 自动生效。

### 2.2 PVC 配置示例

以下是一个生产级别的 PersistentVolumeClaim 配置：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vecboost-models-pvc
  labels:
    app: vecboost
  annotations:
    description: "Shared model storage for VecBoost"
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-regional
  dataSource:
    kind: PersistentVolumeClaim
    name: vecboost-models-source
---
# 模型预热 PVC（初始化时使用）
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vecboost-models-source
  labels:
    app: vecboost
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-regional
```

选择存储类时的建议如下：对于 AWS EKS，推荐使用 `gp3` 或 `io1` 存储类；对于 GKE，推荐使用 `pd-ssd` 存储类；对于自建集群，推荐使用 CephFS 或 NFS 确保跨节点共享访问。

### 2.3 模型预热策略

模型首次加载可能需要 30-60 秒，这会影响首次请求的响应时间。推荐使用模型预热机制：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: vecboost-model-warmup
  labels:
    app: vecboost
spec:
  ttlSecondsAfterFinished: 300
  template:
    metadata:
      labels:
        app: vecboost
    spec:
      restartPolicy: OnFailure
      containers:
      - name: model-warmup
        image: vecboost:latest
        command: ["/app/warmup.sh"])
        volumeMounts:
        - name: model-storage
          mountPath: /models
        env:
        - name: VECBOOST_MODEL_PATH
          value: /models/BAAI/bge-m3
        - name: VECBOOST_WARMUP_ENABLED
          value: "true"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: vecboost-models-pvc
```

预热脚本 `warmup.sh` 的实现逻辑如下：

```bash
#!/bin/bash
set -e

# 等待服务启动
sleep 10)

# 执行预热请求
curl -X POST http://localhost:8080/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "预热请求"}' || true

# 验证模型已加载
if [ -f /models/.loaded ]; then
  echo "Model warmup completed"
  exit 0
else
  echo "Model warmup failed"
  exit 1
fi
```

在 Deployment 中添加 Init Container 确保模型预热完成后再接收流量：

```yaml
spec:
  template:
    spec:
      initContainers:
      - name: model-wait
        image: busybox:1.36
        command: ['sh', '-c', 'while [ ! -f /models/.loaded ]; do sleep 5; done']
        volumeMounts:
        - name: model-storage
          mountPath: /models
      containers:
      - name: vecboost
        image: vecboost:latest
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: vecboost-models-pvc
```

## 3 负载均衡配置

### 3.1 Service 类型选择

根据不同场景选择合适的 Service 类型：

```yaml
# 内部服务发现（推荐用于生产环境）
apiVersion: v1
kind: Service
metadata:
  name: vecboost-internal
  labels:
    app: vecboost
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  selector:
    app: vecboost
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  sessionAffinity: None
```

对于需要对外暴露的服务，使用带有 DNS 名称的 External DNS 配置：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vecboost-public
  labels:
    app: vecboost
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
    external-dns.alpha.kubernetes.io/hostname: "vecboost.example.com"
spec:
  type: LoadBalancer
  selector:
    app: vecboost
  ports:
  - port: 443
    targetPort: 8080
    protocol: TCP
    name: https
```

### 3.2 流量分配策略

对于 GPU 实例和 CPU 实例的混合部署，建议使用不同的 Service 进行流量分配：

```yaml
# GPU 实例专用 Service
apiVersion: v1
kind: Service
metadata:
  name: vecboost-gpu
  labels:
    app: vecboost
    gpu: "true"
spec:
  type: ClusterIP
  selector:
    app: vecboost
    gpu-type: nvidia
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
---
# CPU 实例专用 Service
apiVersion: v1
kind: Service
metadata:
  name: vecboost-cpu
  labels:
    app: vecboost
    gpu: "false"
spec:
  type: ClusterIP
  selector:
    app: vecboost
    gpu-type: cpu
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
---
# 统一入口 Service（自动路由）
apiVersion: v1
kind: Service
metadata:
  name: vecboost
  labels:
    app: vecboost
spec:
  type: ClusterIP
  selector:
    app: vecboost
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
```

通过 Istio VirtualService 实现智能路由：

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vecboost-route
spec:
  hosts:
  - vecboost
  http:
  - match:
    - headers:
        x-latency-sensitive:
          exact: "true"
    route:
    - destination:
        host: vecboost-gpu
      weight: 100
  - route:
    - destination:
        host: vecboost-gpu
      weight: 70
    - destination:
        host: vecboost-cpu
      weight: 30
```

### 3.3 连接池配置

对于高并发场景，配置连接池可以有效管理连接资源：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vecboost-connection-limit
spec:
  podSelector:
    matchLabels:
      app: vecboost
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: vecboost
    ports:
    - protocol: TCP
      port: 8080
```

## 4 监控与告警

### 4.1 Prometheus 指标配置

VecBoost 服务暴露以下关键指标用于监控：

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vecboost-monitor
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: vecboost
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
  namespaceSelector:
    matchNames:
    - vecboost
```

关键指标列表包括以下几个维度。推理性能指标方面，`vecboost_inference_duration_seconds` 记录推理耗时分布，`vecboost_inference_requests_total` 记录推理请求总数，`vecboost_inference_errors_total` 记录推理错误数。资源使用指标方面，`vecboost_memory_usage_bytes` 记录内存使用量，`vecboost_gpu_memory_bytes` 记录 GPU 显存使用量。业务指标方面，`vecboost_embedding_dimensions` 记录向量维度，`vecboost_batch_size` 记录批处理大小。

### 4.2 Grafana 仪表板配置

推荐创建以下 Grafana 仪表板用于监控：

```json
{
  "dashboard": {
    "title": "VecBoost Production Dashboard",
    "panels": [
      {
        "title": "QPS",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vecboost_inference_requests_total[5m])",
            "legendFormat": "QPS"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "P99 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(vecboost_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "P99 Latency"
          }
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vecboost_inference_errors_total[5m]) / rate(vecboost_inference_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "vecboost_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory (GB)"
          }
        ],
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
      },
      {
        "title": "Active Pods",
        "type": "stat",
        "targets": [
          {
            "expr": "count(kube_pod_container_status_ready{namespace='vecboost', pod=~'vecboost-.*'})",
            "legendFormat": "Ready Pods"
          }
        ],
        "gridPos": {"x": 0, "y": 16, "w": 6, "h": 4}
      },
      {
        "title": "GPU Memory",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(vecboost_gpu_memory_bytes / 1024 / 1024 / 1024)",
            "legendFormat": "GPU Memory (GB)"
          }
        ],
        "gridPos": {"x": 6, "y": 16, "w": 6, "h": 4}
      }
    ]
  }
}
```

### 4.3 告警规则配置

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vecboost-alerts
  labels:
    release: prometheus
spec:
  groups:
  - name: vecboost
    rules:
    # 高错误率告警
    - alert: VecBoostHighErrorRate
      expr: |
        rate(vecboost_inference_errors_total[5m]) / rate(vecboost_inference_requests_total[5m]) > 0.01
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "VecBoost error rate exceeds 1%"
        description: "Error rate is {{ $value | humanizePercentage }}"

    # 高延迟告警
    - alert: VecBoostHighLatency
      expr: |
        histogram_quantile(0.99, rate(vecboost_inference_duration_seconds_bucket[5m])) > 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "VecBoost P99 latency exceeds 500ms"
        description: "P99 latency is {{ $value | humanizeDuration }}"

    # 内存不足告警
    - alert: VecBoostHighMemory
      expr: |
        vecboost_memory_usage_bytes / vecboost_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "VecBoost memory usage exceeds 90%"
        description: "Memory usage is {{ $value | humanizePercentage }}"

    # Pod 不可用告警
    - alert: VecBoostPodUnavailable
      expr: |
        kube_pod_container_status_ready{namespace='vecboost', pod=~'vecboost-.*'} == 0
      for: 3m
      labels:
        severity: critical
      annotations:
        summary: "VecBoost pod is unavailable"
        description: "Pod {{ $labels.pod }} has been unavailable for 3 minutes"

    # HPA 无法扩容告警
    - alert: VecBoostHPAMaxReplicas
      expr: |
        kube_hpa_status_desired_replicas{namespace='vecboost', hpa='vecboost-hpa'} >= kube_hpa_spec_max_replicas{namespace='vecboost', hpa='vecboost-hpa'}
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "VecBoost HPA at max replicas"
        description: "HPA has reached max replicas ({{ $value }}) for 10 minutes"
```

### 4.4 日志收集配置

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vecboost-fluentd-config
  labels:
    app: vecboost
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/vecboost/*.log
      pos_file /var/log/vecboost/vecboost.log.pos
      tag vecboost.*
      <parse>
        @type json
        time_key timestamp
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter vecboost.**>
      @type record_transformer
      <record>
        service vecboost
        cluster ${ENV['CLUSTER_NAME']}
        region ${ENV['AWS_REGION']}
      </record>
    </filter>

    <match vecboost.**>
      @type elasticsearch
      host elasticsearch.logging.svc
      port 9200
      index_name vecboost
      type_name _doc
    </match>
```

## 5 性能调优建议

### 5.1 资源配置优化

针对不同工作负载的 Pod 资源配置建议：

```yaml
# CPU 密集型工作负载
spec:
  containers:
  - name: vecboost
    resources:
      requests:
        memory: "4Gi"
        cpu: "2000m"
      limits:
        memory: "6Gi"
        cpu: "4000m"

# 内存密集型工作负载
spec:
  containers:
  - name: vecboost
    resources:
      requests:
        memory: "8Gi"
        cpu: "1000m"
      limits:
        memory: "12Gi"
        cpu: "2000m"

# GPU 工作负载
spec:
  containers:
  - name: vecboost
    resources:
      requests:
        memory: "4Gi"
        cpu: "2000m"
        nvidia.com/gpu: 1
      limits:
        memory: "8Gi"
        cpu: "4000m"
        nvidia.com/gpu: 1
```

资源请求和限制的设置原则如下。CPU 请求应设置为预期平均负载的 1.5 倍，允许短时间突发。CPU 限制应设置为请求的 2 倍，防止单个 Pod 占用过多 CPU。内存请求应设置为平均使用量的 1.2 倍，确保调度时不会 OOM。内存限制应设置为请求的 1.5 倍，提供足够的突发空间。

### 5.2 批处理优化

调整批处理参数以优化吞吐量：

```yaml
env:
- name: VECBOOST_MAX_BATCH_SIZE
  value: "64"
- name: VECBOOST_BATCH_TIMEOUT_MS
  value: "50"
- name: VECBOOST_MAX_CONCURRENT_REQUESTS
  value: "200"
```

批处理参数的选择策略如下。对于 GPU 实例，建议将 `MAX_BATCH_SIZE` 设置为 32-64，充分利用 GPU 并行计算能力。`BATCH_TIMEOUT_MS` 设置为 30-50ms，在延迟和吞吐量之间取得平衡。对于 CPU 实例，`MAX_BATCH_SIZE` 设置为 8-16，避免占用过多 CPU 资源。`MAX_CONCURRENT_REQUESTS` 根据 CPU 核心数设置，建议为 CPU 核心数的 4-8 倍。

### 5.3 网络优化

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vecboost-network-config
data:
  VECBOOST_CONNECTION_BACKLOG: "1024"
  VECBOOST_KEEPALIVE_TIMEOUT: "300"
  VECBOOST_MAX_REQUEST_SIZE: "10485760"
---
# Deployment 中的环境变量配置
spec:
  containers:
  - name: vecboost
    envFrom:
    - configMapRef:
        name: vecboost-network-config
```

网络优化参数说明如下。`CONNECTION_BACKLOG` 设置连接队列大小，建议为 1024-2048。`KEEPALIVE_TIMEOUT` 设置 Keep-Alive 超时时间，单位为秒，建议设置为 300。`MAX_REQUEST_SIZE` 设置最大请求大小，默认为 10MB，适用于长文本处理场景。

### 5.4 内存优化

```yaml
env:
- name: VECBOOST_TOKENIZER_CACHE_SIZE
  value: "1000"
- name: VECBOOST_KV_CACHE_ENABLED
  value: "true"
- name: VECBOOST_MIXED_PRECISION
  value: "true"
```

内存优化策略包括以下几点。Tokenizer 缓存大小根据 QPS 设置，1000 个缓存条目约占用 10-20MB 内存。启用 KV Cache 可以减少重复计算，对于长文本处理效果显著。启用 FP16 混合精度可以将显存占用减少约 50%。

## 6 多区域部署

### 6.1 多区域架构

对于全球化服务，建议采用多区域部署架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     Global Load Balancer                      │
└─────────────────────────────────────────────────────────────┘
                    │                    │
          ┌─────────▼─────────┐  ┌───────▼────────┐
          │  AWS us-east-1    │  │  AWS eu-west-1 │
          │                   │  │                │
          │  ┌─────────────┐  │  │  ┌──────────┐ │
          │  │ VecBoost    │  │  │  │ VecBoost │ │
          │  │ (3 pods)    │  │  │  │ (3 pods) │ │
          │  └─────────────┘  │  │  └──────────┘ │
          │        │          │  │       │       │
          │        ▼          │  │       ▼       │
          │  ┌─────────────┐  │  │  ┌──────────┐ │
          │  │ Model S3    │  │  │  │ Model S3 │ │
          │  │ (Replica)   │  │  │  │ (Replica)│ │
          │  └─────────────┘  │  │  └──────────┘ │
          └───────────────────┘  └───────────────┘
```

### 6.2 跨区域同步配置

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vecboost-sync-config
data:
  VECBOOST_SYNC_REGION: "true"
  VECBOOST_PRIMARY_REGION: "us-east-1"
  VECBOOST_REPLICA_REGIONS: "eu-west-1,ap-southeast-1"
```

## 7 故障恢复策略

### 7.1 Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vecboost-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: vecboost
```

### 7.2 优雅关闭配置

```yaml
spec:
  containers:
  - name: vecboost
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 10 && curl -X POST http://localhost:8080/shutdown"]
    terminationGracePeriodSeconds: 60
```

### 7.3 灾备恢复

```yaml
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: vecboost-backup
  namespace: velero
spec:
  includedNamespaces:
  - vecboost
  storageLocation: default
  ttl: 720h
---
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: vecboost-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"
  template:
    includedNamespaces:
    - vecboost
    storageLocation: default
    ttl: 168h
```

## 8 成本优化

### 8.1 Spot 实例使用

```yaml
spec:
  template:
    spec:
      nodeSelector:
        instance-type: spot
      tolerations:
      - key: "spot-instance"
        operator: "Exists"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: vecboost
              topologyKey: topology.kubernetes.io/zone
```

### 8.2 自动缩容策略

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vecboost-hpa-cost-optimized
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vecboost
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
  behavior:
    scaleDown:
      policies:
      - type: Percent
        value: 50
        periodSeconds: 300
```

## 9 部署检查清单

在生产环境部署前，请确认以下检查项已完成：

基础设施层面需要确认以下内容。存储类已配置且性能满足要求。跨区域网络延迟在可接受范围内。GPU 节点池已正确配置。安全组和网络策略已正确配置。

配置层面需要确认以下内容。ConfigMap 中的所有环境变量已正确设置。Resource limits 已根据实际测试结果调整。HPA 参数已根据业务负载特征优化。Secret 和 TLS 证书已正确配置。

监控层面需要确认以下内容。Prometheus 已配置抓取 VecBoost 指标。Grafana 仪表板已导入。告警规则已配置并测试。日志收集已配置并验证。

运维层面需要确认以下内容。备份策略已配置。故障恢复流程已文档化。容量规划已完成。变更管理流程已就绪。

## 10 常见问题排查

### 10.1 扩容不触发

如果 HPA 不触发扩容，请按以下步骤排查。首先检查指标收集是否正常，Prometheus 中的 `kube_pod_resource_limits` 和 `kube_pod_resource_requests` 指标是否存在。其次确认 HPA 选择器与 Deployment 标签匹配。然后检查 Metrics Server 是否正常运行。最后查看 HPA events 了解具体原因。

### 10.2 模型加载慢

如果模型加载时间过长，请检查以下方面。确认已启用模型预热。确认共享存储性能正常。检查网络带宽是否足够。确认模型文件大小合理。

### 10.3 GPU 利用率低

如果 GPU 利用率不理想，请检查以下方面。增加批处理大小。启用混合精度推理。检查是否有 CPU 瓶颈。优化数据预处理流程。

## 附录

### A. 配置参数速查表

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| VECBOOST_HOST | 0.0.0.0 | 0.0.0.0 | 监听地址 |
| VECBOOST_PORT | 8080 | 8080 | 监听端口 |
| VECBOOST_MODEL_NAME | BAAI/bge-m3 | BAAI/bge-m3 | 模型名称 |
| VECBOOST_DEVICE_TYPE | cpu | cuda | 设备类型 |
| VECBOOST_MAX_BATCH_SIZE | 32 | 32-64 | 最大批处理大小 |
| VECBOOST_MAX_CONCURRENT_REQUESTS | 100 | 100-200 | 最大并发请求数 |
| VECBOOST_TIMEOUT_MS | 60000 | 60000 | 请求超时时间 |
| VECBOOST_TOKENIZER_CACHE_SIZE | 100 | 1000 | Tokenizer 缓存大小 |

### B. 性能基准参考

以下数据在以下环境配置下测得：8 核 CPU、32GB 内存、NVIDIA T4 GPU。

| 指标 | CPU 模式 | GPU 模式 |
|------|----------|----------|
| 单次推理延迟 (P50) | 45ms | 12ms |
| 单次推理延迟 (P99) | 85ms | 28ms |
| 吞吐量 (QPS) | 120 | 850 |
| 内存占用 | 2.5GB | 4GB |
| 显存占用 | N/A | 3.2GB |

### C. 相关资源链接

- VecBoost GitHub 仓库：https://github.com/vecboost/vecboost
- Kubernetes HPA 文档：https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Prometheus Operator：https://github.com/prometheus-operator/prometheus-operator
- Velero 备份工具：https://velero.io/
