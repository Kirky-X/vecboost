# 翻译功能三路评审处置台账

> 记录翻译功能实施工作流中三路独立评审（安全 / 架构 / 性能）共 30 条意见的逐条处置明细。
> 背景：验收时该项因「逐条明细存于工作流 journal、验收员不可读」判 partial；本台账由主会话
> 依据工作流完成通知中的全量 findings 与处置记录整理落盘，处置状态凡标「复核确认」者均经
> 原评审员亲自读代码复核（非采信实现者说明）。
> 后续两处复核说明：
> - `DOC-1` 在工作流内标注「待处置」，归档后主会话实查 `grep -n "panic 作为最后防线" src/text/langdetect.rs`
>   零命中——该条与 `NEW-4`/`RES-1`(b) 为同一问题的重复上报，代码已修复，标注更新为已修复（实查确认）。
> - `PERF-5` 为显性保留（design D2 双层预算防线的设计内成本），非未处置。

## 安全评审（tiangang 方法，5 条）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| SEC-1 | HIGH | `src/text/langdetect.rs:105-116`（panic 点）；`src/service/detect.rs:85-97`、`src/domain/detect.rs:63-72`（放行链） | `/detect` 的 `allow` 传「格式合法但不在 70 发布码表内」的语言码（如 `wo`/`yo`）穿透两道纯格式校验直达 `panic!`，async handler 内无 CatchPanic，认证后远程稳定复现 | 已修复（复核确认）：发布码集合校验前移至入口，未发布码 400 显性拒绝；panic 改 Option 降级不可判定；回归测试钉住 |
| SEC-2 | MEDIUM | `src/engine/onnx_translate.rs:441-453` | 解码 logits 图契约假设无运行时校验（shape 索引/切片越界 panic、debug_assert 仅 debug 生效、NaN expect panic） | 已修复（复核确认）：`validate_logits_shape` 显性校验（含六分支单测，见 NEW-3），NaN 显性报错 |
| SEC-3 | LOW | `src/engine/factory.rs:73-75` | task=translate + candle 拒绝文案内嵌 22 个连续空格 | 已修复（复核确认） |
| SEC-4 | LOW | `src/service/embedding.rs:1966` | 非 onnx 构建下测试代码 `unused variable: disk` 警告 | 已修复（复核确认） |
| SEC-5 | LOW | `src/config/app.rs:65-75`、`src/main.rs:319-323` | config.toml 宣称 `[model]` 可配 task=translate 主模型，但 `translation_params` 无配置入口恒 None，文档与实现静默缺口 | 已修复（复核确认，文档口径修正：该用法参数取缺省，见 `docs/USER_GUIDE.md:534`） |

## 架构评审（diting architecture 方法，9 条）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| ARC-1 | HIGH | `src/text/langdetect.rs:105-116` | 与 SEC-1 同一缺陷的独立发现：allow 非发布码远程必现 panic，文档承诺 400 实际连接重置 | 已修复（复核确认，同 SEC-1 修复） |
| ARC-2 | MEDIUM | `src/main.rs`（pair 循环） | pair 参数缺省 512 硬编码，未复用 `TranslationParams::default()` 单一事实源；`TranslationConfig::validate` 不校验 token 参数 | 已修复（复核确认）：main 与 config 校验均改 default 联动（RES-2 为本条修复的残留复查，亦已修复） |
| ARC-3 | MEDIUM | `src/engine/impl_.rs`（转发臂注释） | 「无覆盖变体经 trait 落默认 UnsupportedTask」注释被无差别粘贴到十余个臂，多数与实际行为相反（count_tokens 实际精确覆盖） | 已修复（复核确认）：逐臂修正注释 |
| ARC-4 | MEDIUM | `src/api/tests.rs` | `forge_translate` handler 装配链单测零覆盖（forge_detect 有，不对称） | 已修复（复核确认）：补 mock 引擎 handler 测试 |
| ARC-5 | LOW | `src/engine/factory.rs` | ConfigError 消息字面量内嵌约 22 连续空格（与 SEC-3 同源） | 已修复（复核确认） |
| ARC-6 | LOW | `src/engine/onnx_translate.rs:452` | argmax NaN 用 expect panic，与同库 NaN 防御先例（candle_decision 容忍 Equal）不一致，第二套手法 | 已修复（复核确认，随 SEC-2 统一为显性报错） |
| ARC-7 | LOW | `src/engine/onnx_translate.rs:371-377` | 引擎层源 token 二次预算校验映射 500，与 design D2「超限冒泡映射 400」文字不符 | 已修复（复核确认） |
| ARC-8 | LOW | `src/service/translate.rs:589-597`、`src/engine/onnx_translate.rs:711-728` | 两个形式主义测试（无断言测试、同义反复恒真测试） | 已修复（复核确认）：删除或替换为有验证价值的断言 |
| ARC-9 | LOW | `src/main.rs` | `translate_service_pair_count` 单行转调包装无抽象价值 | 已修复（复核确认）：内联消除 |

## 性能评审（diting performance 方法，7 条 + 资源 2 条）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| PERF-1 | HIGH | `src/service/translate.rs:157-170`、`src/api/translate.rs:45-52`、`src/main.rs:1697-1710` | 翻译端点无并发背压：同 pair Mutex 串行化，N 并发占 N-1 blocking 线程可饿死全服务共享池；超时后生成循环不可取消白耗算力 | 已修复（复核确认）：`permits=1` 信号量背压 + try_acquire 失败 503 Overloaded，单测钉住（`src/service/translate.rs:140-147`） |
| PERF-2 | MEDIUM | `src/engine/onnx_translate.rs:412-435` | 生成循环每步克隆 encoder 隐状态（≈1-2MB/步）+ 24 次 format! 堆分配 | 已修复（复核确认）：循环不变输入改 `TensorRef::from_array_view` 借用 + 静态名表 |
| PERF-3 | MEDIUM | `src/engine/onnx_translate.rs:449-453` | NaN/空 logits 的 expect panic 位于 Mutex 持有段内，触发即锁中毒、该 pair 永久失效 | 已修复（复核确认）：panic 改显性报错 + `PoisonError::into_inner` 恢复中毒锁 |
| PERF-4 | MEDIUM | `src/engine/onnx_translate.rs:213-223`、`src/main.rs:514-562` | 每 pair 2 Session × 全核 intra 线程，16 对满配 ≈384 常驻线程过订阅 | 已修复（复核确认）：跨 pair 线程预算协调 |
| PERF-5 | LOW | `src/service/translate.rs:141-152`、`src/engine/onnx_translate.rs:363-369` | 热路径对同一文本 tokenize 两次（服务层预判 + 引擎内部 encode） | **显性保留**（评审员复核一致）：双层预算防线为 design D2 明文设计（预判拦截常态超限、引擎兜底防直调穿透），消除重复需改 InferenceEngine trait 签名，毫秒内收益不抵接口变动；P2 pipeline 集成时再评估 |
| PERF-6 | LOW | `src/service/detect.rs:28-33` | 注释「µs 级纯 CPU」对 10k 字符上限文本不成立（静态推断非实测，评审员标注不确定） | 已修复（复核确认）：注释改为如实描述耗时边界 |
| PERF-7 | LOW | `src/config/app.rs:184-199`、`src/main.rs:515-554` | 配置面允许非发布语言码构建死 pair（auto 永远无法命中，静默浪费驻留） | 已修复（复核确认）：入口拒绝非发布码 |
| RES-1(a) | MEDIUM | `src/engine/onnx_translate.rs:380-463`、`src/service/translate.rs:157-170` | 生成循环（最长 1024 步 O(L²)）持锁内执行且 spawn_blocking 无超时 | 已修复（复核确认）：permits=1 背压使排队显性化（与 PERF-1 同向修复）；无取消点语义为设计内已知（见实施报告已知限制 5） |
| RES-2 | LOW | `src/config/app.rs:224-225` | ARC-2 修复残留：validate 内 `unwrap_or(512)` 字面量未复用 default | 已修复（复核确认）：改 `TranslationParams::default()` 联动 |

## 复核轮新增（评审员复核时发现，6 条）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| BUG-1 | CRITICAL | `src/text/langdetect.rs:102-116` + 放行链 | 与 SEC-1/ARC-1 同一缺陷的第三路独立发现（判定 CRITICAL：外部输入稳定触发） | 已修复（复核确认，同 SEC-1 修复） |
| NEW-1 | MEDIUM | `tests/translate_e2e.rs:20-27` | 缺文件级 `#![cfg(feature = "onnx")]` 门控，默认 features 下全量测试编译失败（E0599 ×4） | 已修复（复核确认）：加门控 + 先例注释；前后实测对照（修复前 4×E0599，修复后默认 features cargo check 0 error） |
| NEW-2 | LOW | `src/main.rs:148、514` | 默认 features 构建出现 2 个新警告（unused import / unused mut，cfg 块内消费） | 已修复（复核确认）：cfg 拆分 import + `cfg_attr` allow；实测 `cargo check --bins` 0 警告 |
| NEW-3 | LOW | `src/engine/onnx_translate.rs:101-125` | SEC-2 新增的 `validate_logits_shape` 五分支零单测（校验逻辑自身回归无人能发现） | 已修复（复核确认）：补六分支单测 `test_validate_logits_shape_contract` |
| NEW-4 | LOW | `src/text/langdetect.rs:102-104` | SEC-1 修复残留旧 panic 语义注释与实现矛盾 | 已修复（复核确认）：删除 stale 注释 |
| RES-1(b) | LOW | `src/text/langdetect.rs:102-104` | 与 NEW-4 同一 stale 注释的第二路独立上报 | 已修复（复核确认，同 NEW-4） |

## 文档复核（1 条）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| DOC-1 | LOW | `src/text/langdetect.rs:102-104` | 与 NEW-4/RES-1(b) 同一 stale 注释的第三路上报（工作流内标注「待处置」） | 已修复（归档后主会话实查确认：`grep -n "panic 作为最后防线" src/text/langdetect.rs` 零命中，代码为修复后状态） |

## 汇总

- 30 条意见：**29 条已修复**（其中 28 条经原评审员复核确认，DOC-1 经主会话实查确认），**1 条显性保留**（PERF-5，design D2 设计内成本，非缺陷）。
- CRITICAL/HIGH 共 5 条（SEC-1/ARC-1/BUG-1 同源缺陷三路独立发现 + PERF-1 + RES-1(a) 计 HIGH/MEDIUM 档）：全部修复并复核确认。
- 复核轮未产生「stillOpen」；newFindings（NEW-1..4、RES-1(b)、RES-2）均已闭环。
- 诚实记录：安全评审的 semgrep/trivy 工具扫描层当时未产出有效结果（semgrep 进程崩溃 0 字节输出、trivy vulndb 下载停滞），安全意见以评审员 AI 审查为主——如需工具层补扫可另行执行。
