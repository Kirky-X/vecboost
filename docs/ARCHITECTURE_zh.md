# 架构文档

本文档描述 VecBoost 的内部架构，解释关键组件、数据流和设计决策。

## 目录

- [概述](#概述)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [请求管道](#请求管道)
- [缓存架构](#缓存架构)
- [安全架构](#安全架构)
- [配置系统](#配置系统)
- [性能优化](#性能优化)
- [部署架构](#部署架构)

---

## 概述

VecBoost 是一个使用 Rust 构建的高性能嵌入向量服务。它为文本向量化提供可扩展、生产就绪的解决方案，包含企业级功能。

### 设计目标

1. **高性能**: 通过批处理、并发执行和高效内存管理最小化延迟
2. **可扩展性**: 支持 Kubernetes 水平扩展
3. **可靠性**: 熔断器、重试机制和健康检查
4. **安全性**: 认证、授权和审计日志
5. **可扩展性**: 支持多种推理引擎和部署策略

### 技术栈

| 层级 | 技术 |
|------|------|
| 编程语言 | Rust 2024 Edition |
| Web 框架 | Axum |
| gRPC | Tonic |
| ML 引擎 | Candle（原生 Rust）、ONNX Runtime |
| GPU 支持 | CUDA（NVIDIA）、Metal（Apple） |
| 配置 | TOML |
| 指标 | Prometheus |

---

## 核心组件

### 应用状态

`AppState` 结构体（定义在 `src/lib.rs`）保存路由处理程序使用的所有共享状态：

```rust
pub struct AppState {
    pub service: Arc<RwLock<EmbeddingService>>,
    pub jwt_manager: Option<Arc<JwtManager>>,
    pub user_store: Option<Arc<UserStore>>,
    pub auth_enabled: bool,
    pub csrf_config: Option<Arc<CsrfConfig>>,
    pub csrf_token_store: Option<Arc<CsrfTokenStore>>,
    pub metrics_collector: Option<Arc<InferenceCollector>>,
    pub prometheus_collector: Option<Arc<PrometheusCollector>>,
    pub rate_limiter: Arc<RateLimiter>,
    pub ip_whitelist: Vec<String>,
    pub rate_limit_enabled: bool,
    pub audit_logger: Option<Arc<AuditLogger>>,
    pub pipeline_enabled: bool,
    pub pipeline_queue: Arc<PriorityRequestQueue>,
    pub response_channel: Arc<ResponseChannel>,
    pub priority_calculator: Arc<PriorityCalculator>,
}
```

### 嵌入服务

`EmbeddingService`（`src/service/embedding.rs`）是核心服务，负责协调：

1. **文本处理**: 分块和分词（`src/text/`）
2. **推理**: 通过引擎抽象执行模型（`src/engine/`）
3. **缓存**: 重复查询的结果缓存（`src/cache/`）

```rust
pub struct EmbeddingService {
    engine: Arc<RwLock<AnyEngine>>,
    model_config: Option<ModelConfig>,
    cache: Option<Arc<dyn Cache>>,
    cache_size: usize,
}
```

### 推理引擎

引擎抽象（`src/engine/mod.rs`）为不同的 ML 运行时提供统一接口：

```rust
pub trait Engine: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error>;
    fn get_dimension(&self) -> usize;
    fn health_check(&self) -> bool;
}
```

#### 支持的引擎

| 引擎 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| Candle | 原生 Rust ML 框架 | 无外部依赖，启动快 | 生态系统较小 |
| ONNX Runtime | 跨平台推理 | 成熟，优化良好 | 需要导出 ONNX 模型 |

### 设备管理

设备模块（`src/device/）管理计算设备选择和内存：

```
src/device/
├── mod.rs           # 设备抽象
├── cuda.rs          # CUDA GPU 支持
├── amd.rs           # AMD GPU 支持
├── manager.rs       # 设备生命周期管理
├── memory_pool.rs   # GPU 内存池
├── memory_limit.rs  # 内存限制
└── batch_scheduler.rs  # 批处理优化
```

---

## 数据流

### 请求流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   客户端    │────▶│  HTTP/gRPC  │────▶│   认证和    │────▶│  速率       │
│             │     │   服务器    │     │   安全      │     │  限制       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                      ┌────────────────────────────────────────────┘
                      ▼
              ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
              │  管道       │────▶│  优先级     │────▶│   工作      │
              │  路由器     │     │  计算器     │     │   线程池    │
              └─────────────┘     └─────────────┘     └─────────────┘
                                                         │
                      ┌──────────────────────────────────┘
                      ▼
              ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
              │   缓存      │────▶│  推理       │────▶│   响应      │
              │   检查      │     │   引擎      │     │   构建器    │
              └─────────────┘     └─────────────┘     └─────────────┘
                      │                                       │
                      ▼                                       ▼
              ┌─────────────┐                         ┌─────────────┐
              │  缓存       │◀────────────────────────│    客户端   │
              │   更新      │                         │             │
              └─────────────┘                         └─────────────┘
```

### 逐步处理流程

1. **请求接收**: HTTP/gRPC 服务器接收请求
2. **认证**: JWT 令牌验证（如果启用）
3. **速率限制**: 令牌桶速率限制检查
4. **请求管道**: 可选的优先级队列插入
5. **缓存查找**: 检查缓存的嵌入向量
6. **推理**: 缓存未命中时执行模型
7. **缓存更新**: 存储结果以供将来请求
8. **响应**: 向客户端返回嵌入向量

---

## 请求管道

管道模块（`src/pipeline/）实现基于优先级的请求队列：

```
src/pipeline/
├── mod.rs              # 模块导出
├── config.rs           # 优先级配置
├── priority.rs         # 优先级计算
├── queue.rs            # 线程安全优先级队列
├── scheduler.rs        # 请求调度
├── worker.rs           # 工作线程池
└── response_channel.rs # 异步响应处理
```

### 优先级计算

请求根据多个因素确定优先级：

```rust
pub struct PriorityCalculator {
    base_priority: u32,
    timeout_boost_factor: f32,
    user_tier_weights: HashMap<UserTier, f32>,
    source_weights: HashMap<RequestSource, f32>,
}

impl PriorityCalculator {
    pub fn calculate(&self, request: &PriorityRequest) -> u32 {
        let mut priority = self.base_priority;
        priority += (request.timeout_remaining_secs * self.timeout_boost_factor) as u32;
        priority += (self.user_tier_weights[&request.user_tier] * 100.0) as u32;
        priority += (self.source_weights[&request.source] * 50.0) as u32;
        priority
    }
}
```

### 用户层级权重

| 层级 | 权重 | 描述 |
|------|------|------|
| `free` | 1.0 | 免费层级用户 |
| `basic` | 1.5 | 基础层级用户 |
| `pro` | 2.0 | 专业用户 |
| `enterprise` | 3.0 | 企业客户 |

### 来源权重

| 来源 | 权重 | 描述 |
|------|------|------|
| `api` | 1.0 | API 请求 |
| `grpc` | 1.2 | gRPC 请求（批处理优化） |
| `internal` | 0.5 | 内部服务调用 |

---

## 缓存架构

VecBoost 实现多层缓存系统：

```
src/cache/
├── mod.rs              # 模块导出
├── lru_cache.rs        # LRU 缓存实现
├── lfu_cache.rs        # LFU 缓存实现
├── kv_cache.rs         # 键值缓存
├── arc_cache.rs        # ARC（自适应替换）缓存
└── tiered_cache.rs     # 多层缓存组合
```

### 缓存层次

```
┌─────────────────────────────────────────┐
│           分层缓存                       │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │   ARC   │▶│   LFU   │▶│   KV    │ │
│  │  缓存   │  │  缓存   │  │  缓存   │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│      │             │            │       │
│  频繁访问      长尾        大型       │
│  项目         项目        项目       │
└─────────────────────────────────────────┘
```

### 缓存策略

| 策略 | 用例 | 淘汰策略 |
|------|------|----------|
| **ARC** | 频繁访问的项目 | 自适应 LRU 和 LFU 之间 |
| **LFU** | 一致访问的项目 | 淘汰最少使用 |
| **LRU** | 时间局部性 | 淘汰最近最少使用 |
| **KV** | 大型嵌入向量 | O(1) 键值操作 |

### 缓存配置

```toml
[embedding]
cache_enabled = true
cache_size = 1024  # 最大缓存条目数

[advanced.cache]
# ARC 特定配置
arc_size_fraction = 0.5
# LFU 特定配置
lfu_access_window = 3600
```

---

## 安全架构

### 认证流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   登录      │────▶│   验证      │────▶│   生成      │────▶│   返回      │
│   请求      │     │   密码      │     │   JWT 令牌  │     │   令牌      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   用户      │
                   │   存储      │
                   └─────────────┘
```

### JWT 认证

```rust
pub struct JwtManager {
    key_store: Arc<dyn KeyStore>,
    secret_name: String,
    expiration: Duration,
}

impl JwtManager {
    pub fn generate_token(&self, user_id: &str, roles: &[Role]) -> Result<String, Error> {
        let claims = Claims {
            sub: user_id.to_string(),
            roles: roles.iter().map(|r| r.to_string()).collect(),
            exp: Utc::now() + self.expiration,
            iat: Utc::now(),
        }
        .encode(&self.encoding_key)
    }
}
```

### CSRF 保护

```
src/auth/
├── csrf.rs           # CSRF 令牌生成和验证
├── handlers.rs       # 认证 HTTP 处理程序
├── jwt.rs            # JWT 管理
├── middleware.rs     # Axum 认证中间件
├── mod.rs            # 模块导出
├── types.rs          # 认证类型
└── user_store.rs     # 用户存储
```

### 审计日志

```rust
pub struct AuditLogger {
    log_file: File,
    config: AuditConfig,
}

impl AuditLogger {
    pub async fn log(&self, event: AuditEvent) {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            user_id: event.user_id,
            action: event.action,
            resource: event.resource,
            ip_address: event.ip_address,
            success: event.success,
        };
        // 写入文件
        self.write_entry(&entry).await;
    }
}
```

---

## 配置系统

```
src/config/
├── app.rs            # 应用程序配置
├── model.rs          # 模型配置
└── mod.rs            # 模块导出
```

### 配置层次

1. **默认值**: 代码中的内置默认值
2. **配置文件**: `config.toml` 或 `config_custom.toml`
3. **环境变量**: 覆盖配置值
4. **CLI 参数**: 最高优先级覆盖

### 环境变量映射

| 配置键 | 环境变量 |
|--------|----------|
| `server.port` | `VECBOOST_SERVER_PORT` |
| `model.model_repo` | `VECBOOST_MODEL_REPO` |
| `auth.jwt_secret` | `VECBOOST_JWT_SECRET` |
| `embedding.cache_size` | `VECBOOST_CACHE_SIZE` |

### 配置加载

```rust
impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let mut builder = ConfigBuilder::default();
        
        // 加载配置文件
        builder = builder.add_source(ConfigFile::with_name("config.toml"));
        
        // 添加环境变量
        builder = builder.add_source(EnvironmentVariables::with_prefix("VECBOOST"));
        
        builder.build()
    }
}
```

---

## 性能优化

### 批处理

```
┌─────────────────────────────────────────────────────────┐
│              批处理优化                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  请求 1 ─┐                                              │
│  请求 2 ─┼──▶ 批处理器（最多等待 10ms）──▶              │
│  请求 3 ─┤        [批大小: 32]                          │
│  ...     │              │                               │
│  请求 N ─┘              ▼                               │
│                   ┌──────────────┐                      │
│                   │  批处理      │                      │
│                   │  推理        │                      │
│                   └──────────────┘                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 内存管理

- **内存池**: 预分配的张量缓冲区
- **ArcCache**: 自适应缓存以最小化分配
- **零拷贝**: 尽可能使用共享引用

### GPU 优化

```rust
pub struct MemoryPool {
    buffers: Vec<CudaBuffer>,
    free_list: Vec<usize>,
    max_size: usize,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize) -> Result<CudaBuffer, Error> {
        // 尝试从空闲列表重用
        if let Some(idx) = self.find_free_buffer(size) {
            return Ok(self.buffers[idx].take().unwrap());
        }
        
        // 分配新缓冲区
        self.allocate_new(size)
    }
}
```

### 并发模型

```
                        ┌─────────────────┐
                        │   主线程        │
                        │  (Axum 服务器)  │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  工作线程 1 │    │  工作线程 2 │    │  工作线程 N │
    │ (Rayonpool) │    │ (Rayonpool) │    │ (Rayonpool) │
    └─────────────┘    └─────────────┘    └─────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │  推理           │
                       │  引擎 (GPU)     │
                       └─────────────────┘
```

---

## 部署架构

### Kubernetes 部署

```
deployments/kubernetes/
├── configmap.yaml         # 配置即代码
├── deployment.yaml        # 主部署
├── gpu-deployment.yaml    # GPU 节点选择器
├── hpa.yaml               # 水平 Pod 自动扩缩容
├── model-cache.yaml       # 模型存储 PVC
├── service.yaml           # 集群 IP 服务
└── SCALING_BEST_PRACTICES.md
```

### 容器架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker 容器                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │  VecBoost 进程 (PID 1)                               │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────────────────┐   │    │
│  │  │ HTTP    │ │ gRPC    │ │ 健康检查            │   │    │
│  │  │ 服务器  │ │ 服务器  │ │ /health             │   │    │
│  │  └────┬────┘ └────┬────┘ └─────────────────────┘   │    │
│  └───────┼───────────┼──────────────────────────────────┘    │
│          │           │                                       │
│          ▼           ▼                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              推理引擎                                │    │
│  │         (Candle / ONNX Runtime)                      │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │                                   │
│          ┌────────────────┼────────────────┐                 │
│          ▼                ▼                ▼                 │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│   │   CPU      │  │   CUDA     │  │   Metal    │            │
│   │  内存      │  │  VRAM      │  │   VRAM     │            │
│   └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 扩展策略

| 策略 | 描述 | 用例 |
|------|------|------|
| **水平 Pod 自动扩缩容** | 基于 CPU/内存添加副本 | 高请求量 |
| **GPU 节点池** | 专用 GPU 节点 | 推理密集型工作负载 |
| **模型缓存** | 持久化存储模型 | 多区域部署 |
| **速率限制** | 防止过载 | 公共 API |

---

## 扩展点

### 添加新引擎

1. 在 `src/engine/` 实现 `Engine` trait
2. 将引擎类型添加到 `EngineType` 枚举
3. 更新 `AnyEngine::new()` 工厂方法
4. 添加配置解析

```rust
pub trait Engine: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error>;
    fn get_dimension(&self) -> usize;
    fn health_check(&self) -> bool;
}
```

### 添加新缓存

1. 在 `src/cache/` 实现 `Cache` trait
2. 将缓存类型添加到 `CacheType` 枚举
3. 更新 `EmbeddingService` 中的缓存工厂

### 自定义认证

1. 实现 `AuthProvider` trait
2. 在认证模块注册
3. 在 `config.toml` 中配置

---

## 错误处理

```
src/error.rs
```

### 错误类型

| 错误 | 描述 | 恢复策略 |
|------|------|----------|
| `InferenceError` | 模型推理失败 | 指数退避重试 |
| `CacheMiss` | 缓存条目未找到 | 回退到推理 |
| `RateLimitExceeded` | 触发速率限制 | 等待后重试 |
| `CircuitBreakerOpen` | 熔断器打开 | 快速失败，等待恢复 |
| `GPUOutOfMemory` | GPU 内存耗尽 | 回退到 CPU |
| `ModelNotFound` | 模型不可用 | 下载或切换模型 |
