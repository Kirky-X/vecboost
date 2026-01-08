#!/bin/bash

# VecBoost 部署和监控脚本
# 用于在测试环境部署服务并监控运行情况

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
IMAGE_NAME="vecboost:latest"
CONTAINER_NAME="vecboost-test"
HOST_PORT=8080
METRICS_PORT=9090
GRPC_PORT=50051
LOG_DIR="./logs"
MODEL_DIR="./models"
CACHE_DIR="./cache"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    print_success "Docker 已安装: $(docker --version)"
}

# 检查必要的目录
create_directories() {
    print_info "创建必要的目录..."
    mkdir -p "$LOG_DIR" "$MODEL_DIR" "$CACHE_DIR"
    print_success "目录创建完成"
}

# 构建 Docker 镜像
build_image() {
    print_info "开始构建 Docker 镜像..."
    docker build -t "$IMAGE_NAME" . || {
        print_error "Docker 镜像构建失败"
        exit 1
    }
    print_success "Docker 镜像构建完成"
}

# 停止并删除旧容器
stop_old_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "停止旧容器..."
        docker stop "$CONTAINER_NAME" || true
        docker rm "$CONTAINER_NAME" || true
        print_success "旧容器已删除"
    fi
}

# 启动容器
start_container() {
    print_info "启动新容器..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${HOST_PORT}:8080" \
        -p "${METRICS_PORT}:9090" \
        -p "${GRPC_PORT}:50051" \
        -v "$(pwd)/${LOG_DIR}:/app/logs" \
        -v "$(pwd)/${MODEL_DIR}:/app/models" \
        -v "$(pwd)/${CACHE_DIR}:/app/cache" \
        -e RUST_LOG=vecboost=debug \
        -e VECBOOST_HOST=0.0.0.0 \
        -e VECBOOST_PORT=8080 \
        -e VECBOOST_MEMORY_POOL_ENABLED=true \
        -e VECBOOST_PIPELINE_ENABLED=true \
        "$IMAGE_NAME" || {
        print_error "容器启动失败"
        exit 1
    }
    print_success "容器启动成功"
}

# 等待服务就绪
wait_for_service() {
    print_info "等待服务就绪..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf http://localhost:${HOST_PORT}/health > /dev/null 2>&1; then
            print_success "服务已就绪"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    print_error "服务启动超时"
    return 1
}

# 检查容器状态
check_container_status() {
    print_info "检查容器状态..."
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# 查看容器日志
view_logs() {
    print_info "最近 50 行日志:"
    docker logs --tail 50 "$CONTAINER_NAME"
}

# 测试健康检查端点
test_health_endpoint() {
    print_info "测试健康检查端点..."
    local response=$(curl -s http://localhost:${HOST_PORT}/health 2>/dev/null || echo "{}")
    if echo "$response" | grep -q "healthy"; then
        print_success "健康检查通过"
        echo "响应: $response"
    else
        print_warning "健康检查响应: $response"
    fi
}

# 测试 Prometheus 指标
test_metrics_endpoint() {
    print_info "测试 Prometheus 指标端点..."
    local response=$(curl -s http://localhost:${METRICS_PORT}/metrics 2>/dev/null || echo "")
    if [ -n "$response" ]; then
        print_success "Prometheus 指标端点正常"
        echo "指标数量: $(echo "$response" | wc -l)"
    else
        print_warning "Prometheus 指标端点无响应"
    fi
}

# 运行性能测试
run_performance_test() {
    print_info "运行性能测试..."

    # 测试单个请求延迟
    print_info "测试单个请求延迟..."
    local start=$(date +%s%N)
    curl -s -X POST http://localhost:${HOST_PORT}/api/v1/embed \
        -H "Content-Type: application/json" \
        -d '{"text":"性能测试文本","normalize":true}' > /dev/null
    local end=$(date +%s%N)
    local latency=$(( (end - start) / 1000000 ))
    print_success "单个请求延迟: ${latency}ms"

    # 测试并发请求
    print_info "测试并发请求（10 个并发）..."
    local start=$(date +%s%N)
    for i in {1..10}; do
        curl -s -X POST http://localhost:${HOST_PORT}/api/v1/embed \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"并发测试文本${i}\",\"normalize\":true}" > /dev/null &
    done
    wait
    local end=$(date +%s%N)
    local total_time=$(( (end - start) / 1000000 ))
    local qps=$(( 10000 / total_time ))
    print_success "10 个并发请求完成，总耗时: ${total_time}ms，QPS: ${qps}"
}

# 监控容器资源使用
monitor_resources() {
    print_info "容器资源使用情况:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" "$CONTAINER_NAME"
}

# 实时监控
realtime_monitor() {
    print_info "开始实时监控（按 Ctrl+C 退出）..."
    print_info "监控间隔: 5 秒"

    while true; do
        echo -e "\n========================================"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"

        # 容器状态
        echo -e "\n【容器状态】"
        check_container_status

        # 资源使用
        echo -e "\n【资源使用】"
        monitor_resources

        # 健康检查
        echo -e "\n【健康检查】"
        if curl -sf http://localhost:${HOST_PORT}/health > /dev/null 2>&1; then
            print_success "服务健康"
        else
            print_warning "服务不健康"
        fi

        # 最近的日志
        echo -e "\n【最近的日志】"
        docker logs --tail 10 "$CONTAINER_NAME" 2>&1 | tail -5

        sleep 5
    done
}

# 快速监控
quick_monitor() {
    print_info "快速监控（30 秒）..."

    for i in {1..6}; do
        echo -e "\n监控轮次 $i/6:"
        monitor_resources
        test_health_endpoint
        sleep 5
    done
}

# 清理
cleanup() {
    print_info "清理资源..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    print_success "清理完成"
}

# 显示菜单
show_menu() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "   VecBoost 部署和监控工具"
    echo "========================================"
    echo "   1. 完整部署（构建 + 启动 + 监控）"
    echo "   2. 仅构建镜像"
    echo "   3. 仅启动容器"
    echo "   4. 测试健康检查端点"
    echo "   5. 测试 Prometheus 指标端点"
    echo "   6. 运行性能测试"
    echo "   7. 快速监控（30 秒）"
    echo "   8. 实时监控（按 Ctrl+C 退出）"
    echo "   9. 清理容器"
    echo "   0. 退出"
    echo "========================================"
    echo -e "${NC}"
}

# 主函数
main() {
    check_docker
    create_directories

    # 解析命令行参数
    if [ $# -gt 0 ]; then
        case "$1" in
            deploy)
                build_image
                stop_old_container
                start_container
                wait_for_service
                check_container_status
                test_health_endpoint
                ;;
            monitor)
                realtime_monitor
                ;;
            cleanup)
                cleanup
                ;;
            *)
                print_error "未知命令: $1"
                exit 1
                ;;
        esac
        exit 0
    fi

    # 交互式菜单
    while true; do
        show_menu
        echo -n "请输入选项（0-9）: "
        read choice

        case $choice in
            1)
                print_info "开始完整部署..."
                build_image
                stop_old_container
                start_container
                wait_for_service
                check_container_status
                test_health_endpoint
                print_info "部署完成，选择 7 或 8 开始监控"
                ;;
            2)
                build_image
                ;;
            3)
                stop_old_container
                start_container
                wait_for_service
                check_container_status
                ;;
            4)
                test_health_endpoint
                ;;
            5)
                test_metrics_endpoint
                ;;
            6)
                run_performance_test
                ;;
            7)
                quick_monitor
                ;;
            8)
                realtime_monitor
                ;;
            9)
                cleanup
                ;;
            0)
                print_info "退出"
                exit 0
                ;;
            *)
                print_error "无效的选项"
                ;;
        esac

        echo ""
        read -p "按 Enter 继续..."
    done
}

# 运行主函数
main "$@"