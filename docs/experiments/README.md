# vecboost 实验协议（A/B 实证纪律）

> Port 自 colibri 的 experiment manifest 协议：一条优化在拿到受控 A/B 数据之前只是假设。
> 校验器：`python3 scripts/validate_manifest.py <manifest.json>`（仅标准库）。

## 硬规则（四条）

1. **单变量**：一次实验只改一个变量（构建 flag / 配置项 / 环境变量）。改了两个变量 = 结论作废。
2. **次数与中位数**：baseline 与 trial 各 **≥3 次**，报**中位数**而非单次最佳；两次实验之间须预热（至少跑一轮弃掉）以稳定页缓存与分支预测。
3. **ABBA 顺序**：按 `baseline→trial→trial→baseline`（或更长的回文序）交替执行，抵抗机器热态/页缓存/后台负载的单向漂移。colibri 有过真实教训：跨进程测得 +7.1% 的优化，改用集成 ABBA 后实测 **-18.8%**，结论被推翻。
4. **吞吐不得搬移时间**：decode/查询吞吐的提升不得以把开销搬移到 startup/prefill/首次请求为代价——报告必须同时给出 TTFT（或首请求延迟）与稳态吞吐；无损优化（如批内去重、时间窗拼批）还必须证明输出与基线**字节等同**。

## 每类优化的验收面

| 优化类型 | 必须证明 |
|---|---|
| 无损（拼批/去重/线程/调度） | 输出字节等同 + 吞吐或 p99 改善 ≥3% |
| 有损（量化） | golden 语料质量门（余弦中位数阈值）+ 体积/吞吐收益 |
| 内存/驻留（LFRU/WAL） | 驻留上限生效 + 命中率不降 + 驱逐不产生错误响应 |

## manifest 字段

参考 `manifest.template.json`。必填：

- `hypothesis` — 假设一句话（"X 因 Y 而 提升 Z"）
- `change.commit` / `change.variable` — 被测 commit 与**唯一**变量
- `hardware` — CPU 型号/物理核/内存/OS（性能数字必须标注主机）
- `baseline` / `trial` — 各含 `runs: [数值, ...]`（≥3）与 `median`
- `quality` — 无损: `byte_identical: true`；有损: 门值与实测中位数
- `evidence` — 原始日志文件路径 + 该文件 sha256

校验器检查：必填字段齐全、runs ≥3、声明 median 与重算一致、evidence sha256 一致。**校验通过 ≠ 结论成立**——ABBA 顺序与单变量由评审人核对。

## 流程

1. 建 `docs/experiments/<日期>-<slug>.manifest.json`（复制模板）
2. 跑 baseline×3 与 trial×3（ABBA 顺序），原始输出存 `evidence` 并算 sha256
3. `python3 scripts/validate_manifest.py <manifest>` → 通过后连同 evidence 提交
4. 结论回写 `docs/PERFORMANCE.md` 对应条目（正反结果都记——负结果同样有价值；原 `docs/tuning.md` 已并入该文件）
