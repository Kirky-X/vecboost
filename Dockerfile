# VecBoost 生产环境 Dockerfile
# 多阶段构建，优化镜像大小和安全性

# ============================================
# 阶段 1: 构建阶段
# ============================================
FROM rust:1.75-slim as builder

# 设置工作目录
WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制 Cargo 配置
COPY Cargo.toml Cargo.lock ./

# 创建虚拟 main.rs 以预编译依赖（利用 Docker 缓存）
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# 复制源代码
COPY src ./src
COPY proto ./proto
COPY build.rs ./

# 构建 Release 版本（启用所有特性）
RUN cargo build --release --features cuda,onnx,grpc

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
RUN mkdir -p /app/models /app/logs /app/cache \
    && chown -R vecboost:vecboost /app

# 设置工作目录
WORKDIR /app

# 从构建阶段复制二进制文件
COPY --from=builder /build/target/release/vecboost /app/vecboost
COPY config.toml /app/config.toml

# 设置权限
RUN chmod +x /app/vecboost && \
    chown vecboost:vecboost /app/vecboost /app/config.toml

# 切换到非 root 用户
USER vecboost

# 暴露端口
EXPOSE 8080 9090 50051

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 设置环境变量
ENV RUST_LOG=vecboost=info \
    VECBOOST_HOST=0.0.0.0 \
    VECBOOST_PORT=8080 \
    VECBOOST_MODEL_PATH=/app/models \
    VECBOOST_LOG_PATH=/app/logs \
    VECBOOST_CACHE_PATH=/app/cache

# 启动应用
CMD ["/app/vecboost"]