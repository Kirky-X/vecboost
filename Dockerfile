# VecBoost 生产环境 Dockerfile(edition 2024;生态库 MSRV confers/limiteron rc.5
# 需 rustc ≥ 1.97.1,故基础镜像取 rust:1.98,与 CI stable 工具链同版)
#
# 多阶段构建,优化镜像大小和安全性。
#
# 构建上下文约定:本仓库的跨仓 path 依赖(trait-kit、confers、inklog、oxcache、
# limiteron、dbnexus、sdforge)已全部切换 crates.io 发布版,不再需要同级 base/
# 生态库目录,构建 context 即本仓库根目录(CI docker.yml 的 context: . 一致):
#
#   docker build -t vecboost:latest .
#
# 默认构建不启用 cuda(镜像内无 NVIDIA 工具链);需要 GPU 时请基于
# nvidia/cuda 镜像自建并追加 --features cuda,onnx。

# ============================================
# 阶段 1: 构建阶段
# ============================================
# bookworm 变体:与运行时镜像 debian:bookworm-slim 的 libssl/glibc ABI 一致
FROM rust:1.98-slim-bookworm AS builder

# 设置工作目录
WORKDIR /build

# 安装构建依赖
# make:tikv-jemalloc-sys 的 configure 需要;curl:utoipa-swagger-ui build.rs
# 经系统 curl 下载 swagger-ui 资源(slim 镜像均不预装)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制清单以预编译依赖(利用 Docker 缓存);workspace 成员 examples 的
# 清单一并复制,否则 cargo 因 workspace member 清单缺失拒绝加载
COPY Cargo.toml Cargo.lock build.rs ./
COPY examples/Cargo.toml ./examples/

# 根清单显式声明 4 个 [[bench]] 目标(harness=false),清单解析要求文件存在:
# stub 阶段以同名空 main 占位(真实 bench 不参与 -p vecboost 构建,仅占位)
RUN mkdir -p src examples/src benches && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > examples/src/main.rs && \
    for b in similarity_bench batch_scheduling_bench semantic_cache_bench embed_throughput_bench; do \
        echo "fn main() {}" > "benches/$b.rs"; \
    done && \
    cargo build --release -p vecboost && \
    rm -rf src examples/src benches

# 复制源代码与资源;examples/Cargo.toml 由上一层的 COPY 保留(workspace 加载必需),
# benches 补回真实实现([[bench]] 目标文件存在是清单解析的硬性要求)
COPY src ./src
COPY benches ./benches
COPY config ./config

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
COPY --from=builder /build/target/release/vecboost /app/vecboost
COPY --from=builder /build/config/config.toml /app/config/config.toml

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
