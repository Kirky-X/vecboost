# VecBoost Grafana 仪表板

`vecboost-dashboard.json` 为预配置仪表板（Grafana 10+），覆盖 HTTP、批处理、队列、引擎阶段、语义缓存、决策、检测与翻译指标。所有数据来自服务 `/metrics`（Prometheus 文本格式）。

## 前置

1. 服务以 HTTP 面运行，且在 Prometheus 侧抓取 `/metrics`（默认端口见 `[server] port`）。
2. Prometheus 数据源已加入 Grafana（Configuration → Data sources → Prometheus）。

## 导入

1. Grafana 左侧 **Dashboards → New → Import**。
2. 选择 `deploy/grafana/vecboost-dashboard.json`（或粘贴内容）。
3. 在 `Prometheus` 下拉中选择目标数据源（对应模板变量 `DS_PROMETHEUS`）。
4. 点击 **Import**。

## 面板与指标口径

| 面板 | 指标 |
|------|------|
| HTTP RPS / P95 延迟 | `http_requests_total`、`http_request_duration_seconds_bucket` |
| 批大小 / 批等待 / 批内去重比 | `vecboost_batch_size_*`、`vecboost_batch_wait_seconds_bucket`、`vecboost_inbatch_dedup_ratio` |
| 队列深度 / 在途请求 / 队列等待 | `vecboost_pipeline_queue_depth`、`vecboost_pipeline_in_flight_requests`、`vecboost_queue_wait_seconds_bucket` |
| 阶段耗时 | `vecboost_stage_seconds_bucket{stage=tokenize\|inference\|pool}` |
| 语义缓存模糊命中 | `vecboost_semantic_cache_fuzzy_hits_total` |
| 决策延迟 / 决策请求 | `vecboost_decision_seconds_bucket`、`vecboost_decision_requests_total` |
| 检测请求 / 翻译请求 | `vecboost_detect_requests_total`、`vecboost_translate_requests_total` |

延迟类面板使用 `histogram_quantile(0.95, ...)` 展示 P95；批大小为 `_sum/_count` 的均值。指标定义与调优开关见 [性能指南](../../docs/PERFORMANCE.md) 与 [用户指南](../../docs/USER_GUIDE.md)。
