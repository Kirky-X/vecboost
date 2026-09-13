# VecBoost 生产环境 Dockerfile(edition 2024,需 Rust ≥ 1.85)
#
# 多阶段构建,优化镜像大小和安全性。
#
# 构建上下文约定:vecboost 通过路径依赖引用同级 `base/` 生态库(trait-kit、
# confers、inklog、oxcache、limiteron、dbnexus、sdforge),因此构建 context
# 必须是同时包含两个仓库的目录。在工作区父目录执行:
#
#   docker build -f vecboost/Dockerfile -t vecboost:latest .
#
# 目录布局要求:
#   <context>/
#     ├── vecboost/   # 本仓库
#     └── base/       # 生态库集合(trait-kit/ confers/ inklog/ ...)
#
# 默认构建不启用 cuda(镜像内无 NVIDIA 工具链);需要 GPU 时请基于
# nvidia/cuda 镜像自建并追加 --features cuda,onnx。

# ============================================
# 阶段 1: 构建阶段
# ============================================
FROM rust:1.85-slim AS builder

# 设置工作目录
WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制清单(含路径依赖的 base/ 生态库)以预编译依赖(利用 Docker 缓存)
COPY vecboost/Cargo.toml vecboost/Cargo.lock vecboost/build.rs ./vecboost/
COPY base ./base

WORKDIR /build/vecboost
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release -p vecboost && \
    rm -rf src

# 复制源代码与资源
COPY vecboost/src ./src
COPY vecboost/config ./config

# 构建 Release 版本(默认 HTTP 特性;按需追加 --features)
RUN touch src/main.rs && cargo build --release -p vecboost --features http

# ============================================
# 阶段 2: 运行阶段
# ============================================
FROM debian:bookworm-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r vecboost && useradd -r -g vecboost vecboost

# 创建必要的目录
RUN mkdir -p /app/models /app/logs /app/cache /app/config \
    && chown -R vecboost:vecboost /app

# 设置工作目录
WORKDIR /app

# 从构建阶段复制二进制文件与出厂配置
COPY --from=builder /build/vecboost/target/release/vecboost /app/vecboost
COPY --from=builder /build/vecboost/config/config.toml /app/config/config.toml

# 设置权限
RUN chmod +x /app/vecboost && \
    chown vecboost:vecboost /app/vecboost /app/config/config.toml

# 切换到非 root 用户
USER vecboost

# 暴露端口(与出厂配置 config.toml 的 server.port=9002 一致;50051 为 gRPC 可选)
EXPOSE 9002 50051

# 健康检查(轻量 liveness;就绪探测用 /health?depth=full)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9002/health || exit 1

# 设置环境变量
# 安全默认:启用认证时必须提供 VECBOOST_JWT_SECRET 与 VECBOOST_ADMIN_PASSWORD。
# 容器需监听 0.0.0.0 以便端口映射;出厂配置 auth.enabled=false 会触发启动闸门,
# 故镜像默认携带 VECBOOST_ALLOW_INSECURE=1(启动时打 ERROR 告警)。
# 生产部署请挂载开启 auth 的配置(或以环境变量覆盖)并删除本逃生阀,
# 访问控制由宿主防火墙/反代 + 应用层认证共同承担。
ENV RUST_LOG=vecboost=info \
    VECBOOST_HOST=0.0.0.0 \
    VECBOOST_PORT=9002 \
    VECBOOST_ALLOW_INSECURE=1 \
    VECBOOST_MODEL_PATH=/app/models \
    VECBOOST_LOG_PATH=/app/logs \
    VECBOOST_CACHE_PATH=/app/cache

# 启动应用
CMD ["/app/vecboost"]
