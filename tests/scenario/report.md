# VecBoost 国内使用场景全量测试报告 — full-scenario-testing

- **执行日期**：2026-09-04（WSL2 + RTX 5080 16GB + CUDA 12.8 + Rust 1.97.1）
- **被测版本**：dev @ 57bd05b（Merge branch 'dev'），全特性 debug 构建（http,grpc,cli,auth,db,mcp）
- **测试模型**：M1 = BAAI/bge-small-en-v1.5（BERT/384 维/本地加载）；M2 = BAAI/bge-small-zh-v1.5（XLM-R/512 维/服务端 HF 下载）
- **执行方式**：`bash scripts/run-scenario-tests.sh`（54 个 pytest 用例 + CLI/MCP/library 模式探针）
- **场景规格**：specmark/changes/full-scenario-testing/specs/（6 能力域 × 正常/异常，MECE 编号）

## 一、套件结果总览

| 套件 | 用例数 | 通过 | 失败 | 跳过 | 红灯归因 |
|------|--------|------|------|------|----------|
| test_embed_normal | 8 | 6 | 1 | 1 | DEFECT-SIM-002（相似度恒 1.0） |
| test_embed_abnormal | 11 | 7 | 2 | 2 | DEFECT-VAL-001（校验错误→500） |
| test_rerank | 6 | 5 | 1 | 0 | DEFECT-RERANK-001（top_k=0 未校验） |
| test_model_zh | 7 | 6 | 0 | 1 | 全绿（R-model-007 能力缺口 skip） |
| test_auth | 8 | 4 | 0 | 4 | skip 均因 DEFECT-AUTH-001（登录不可达） |
| test_security | 6 | 3 | 1 | 2 | DEFECT-RL-001（限流未生效） |
| test_server_modes | 9 | 4 | 3 | 2 | DEFECT-METRICS/SHUTDOWN/CLI/LIB-001 |
| test_config_device | 6 | 5 | 0 | 1 | 全绿（R-config-003 环境受限 skip） |
| **合计** | **54+** | **40** | **8** | **12** | 红灯 100% 对应已坐实缺陷，0 来因测试自身 |

> 模式探针：MCP ✓（initialize + 16 工具 + tools/call embed_text 返回 384 维）；CLI ✗（DEFECT-CLI-001）；library 示例 ✗（DEFECT-LIB-001）。

## 二、产品缺陷清单（12 项，按严重度）

| ID | 严重度 | 描述 | 触达场景 |
|----|--------|------|----------|
| DEFECT-AUTH-001 | HIGH | auth 白名单前缀 /api/v1 ≠ 实际路由 /api/1，登录被自身中间件拦截，**认证体系整体不可用** | R-auth-001~004、007 |
| DEFECT-GRPC-001 | HIGH | grpc_enabled=true 主流程在 sdforge LimiteronAdapter（Governor 构建处）永久阻塞，HTTP/gRPC 均不服务，进程僵死 | R-server-003 |
| DEFECT-SIM-002 | HIGH | /similarity 无关文本对返回 1.0，相似度计算结果错误 | R-embed-003 |
| DEFECT-RL-001 | HIGH | 限流未生效（阈值 6/min + 空白名单十连发全 200），限流器疑似未接入 HTTP 管线 | R-auth-009 |
| DEFECT-HUB-001 | HIGH（国内） | hf-hub 对 hf-mirror.com 报 missing ETag header，服务端模型下载国内镜像不可用（curl 直拉正常） | R-model-002 |
| DEFECT-VAL-001 | MED | 超长文本/超批量校验错误返回 500（应 400/422） | R-embed-009 |
| DEFECT-SHUTDOWN-001 | MED | SIGTERM 优雅关闭挂死（>35s 需 SIGKILL），影响容器滚动升级 | R-server-001 |
| DEFECT-METRICS-001 | MED | /metrics 返回 200 空 body，Prometheus 导出未生效 | R-server-007 |
| DEFECT-AUDIT-001 | MED | audit.enabled=true 下 audit.log 从未创建，安全事件零落盘 | R-auth-011 |
| DEFECT-CLI-001 | MED | CLI `embed --req` exit 0 但无任何结果输出（stdout 全为启动日志） | R-server-004 |
| DEFECT-LIB-001 | LOW | library_usage 示例崩溃（async runtime 内 block_on，退出码 101）；library 模块本身未验证 | R-server-006 |
| DEFECT-SWITCH-001 | LOW | 无效模型切换返回 500（应 4xx）；原模型韧性正常 | R-model-005、R-auth-010 |

## 三、能力边界（非缺陷）

1. `embed/search` 未暴露 HTTP 路由（服务层有实现）。
2. `SimilarityRequest` 无 metric 字段——4 种度量不可经 API 选择。
3. 无 `model/unload` HTTP 路由。
4. `/model/info` 无 device 字段（设备仅启动日志可见）。
5. `download_model --small` 下载 safetensors 稳定失败（5/6），需 curl 经镜像补齐。
6. CLI/MCP 工具入参为 `{"req": <DTO>}` 包装签名（与常见 --text 风格不同，文档需说明）。

## 四、国内部署要点结论

1. **镜像不可用是硬伤**：hf-hub 1.0.0 与 hf-mirror.com 不兼容（DEFECT-HUB-001），国内无外网环境只能走本地模型目录加载（该路径验证 ✓ 完整可用）。
2. **认证与 gRPC 当前不可用**（AUTH-001/GRPC-001），生产暴露前必须修复。
3. **优雅关闭挂死**影响 K8s 滚动更新（SHUTDOWN-001）。
4. CPU 模式全功能可用（含中文模型），无 GPU 环境可正常部署（R-config-005 ✓）。
5. 限流/审计/指标三大运维面均未生效（RL-001/AUDIT-001/METRICS-001），上生产前需修复。

## 五、模型矩阵覆盖（用户要求：嵌入与重排各 ≥2 模型）

| 模型 | 架构/维度 | 加载方式 | 嵌入场景 | 重排场景 |
|------|-----------|----------|----------|----------|
| BAAI/bge-small-en-v1.5 | BERT / 384 | 本地目录 | ✓ 全部（中英混合/emoji/批量/Matryoshka/OpenAI 兼容） | ✓（排序/分数域/top_k） |
| BAAI/bge-small-zh-v1.5 | XLM-R / 512 | HF 下载（直连回退） | ✓（简体中文 512 维） | ✓（中文相关性排序） |

## 六、服务器模式覆盖（用户要求）

| 模式 | 结果 | 备注 |
|------|------|------|
| HTTP | ✓ 全端点（嵌入/重排/相似度/文件/模型管理/OpenAI 兼容/health） | 唯一完整可用模式 |
| gRPC | ✗ DEFECT-GRPC-001 | 无法 E2E |
| CLI | 部分（--req 签名可用、exit 0）| 结果无输出（DEFECT-CLI-001） |
| MCP | ✓ 完整（16 工具，embed_text 实测返回向量） | stdio JSON-RPC |
| 嵌入式 library | ✗ DEFECT-LIB-001 | 示例崩溃，模块本身未验证 |

## 七、复现方式

```bash
cargo build --features "http,grpc,cli,auth,db,mcp"
bash scripts/run-scenario-tests.sh --skip-build
# 单套件：python3 -m pytest tests/scenario/test_embed_normal.py -v
```
