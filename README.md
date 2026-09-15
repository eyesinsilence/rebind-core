# ReBind Core

ReBind RAG 研究实现的代码版本，包括来源重读、联合变量绑定、问题图执行、训练与评测接口。

本仓库只包含 Python 源码、配置模板和测试。实验数据、标签、检索索引、模型权重、检查点、预测、日志、历史清单及第三方仓库均需另行准备。

## 核心模块

| 模块 | 功能 |
|---|---|
| `src/rebind_mvp/source_reader.py` | 来源注意力、来源重读与 ReBind 模块 |
| `src/rebind_mvp/pair_update.py` | 候选匹配的三角消息更新 |
| `src/rebind_mvp/bp.py` | belief propagation 对照 |
| `src/rebind_mvp/joint_decoder.py` | 带约束的联合候选解码 |
| `src/rebind_mvp/schema.py` | 问题图、候选注册表与证据存储 |
| `src/rebind_mvp/final_frontend.py` | 问题编译、候选来源跟踪和依赖调度 |
| `src/rebind_mvp/final_alignment.py` | 部分标签构建、对齐训练和监督损失 |
| `src/rebind_mvp/adapt.py` | 批处理 ReBind 与适配训练 |
| `src/rebind_mvp/transitions.py` | 来源窗口特征和轮次间状态更新 |
| `src/rebind_mvp/evaluate.py` | 检索与回答循环 |
| `src/rebind_mvp/proposal.py` / `retrieval.py` | 生成器、候选提议及 E5 检索接口 |
| `scripts/run_final_plan.py` | 最终实验入口与检查点身份验证 |

包内同时保留上述模块依赖的数据适配器、诊断函数和基线适配接口；没有附带相应数据。

## 安装与测试

推荐 Python 3.12。先按运行机器选择适合的 PyTorch 版本，再在仓库根目录安装：

```bash
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -e ".[test]"
python -m pytest -q
```

测试使用代码内构造的小型张量和占位实体。两项官方 MQuAKE 接口测试需要自行准备 `upstream/mquake_remastered/`，缺失时跳过；sandbox 测试需要 Linux 的 `bubblewrap`，缺失时跳过。核心算子测试不需要下载模型或数据。

可直接导入核心模块：

```python
from rebind_mvp.source_reader import ReBindModule
from rebind_mvp.joint_decoder import decode
from rebind_mvp.final_frontend import validate_graph, next_action

model = ReBindModule(input_dim=768, d=128, layers=4, mode="rebind")
```

`ReBindModule.forward` 的输入张量形状和来源关联掩码见 `source_reader.py`，完整调用示例见 `tests/test_core_integration.py`。

## 接入真实实验

1. 自行提供模型目录、数据、索引及所需的第三方基线代码。
2. 编辑 `configs/final.yaml` 中的 `paths`、`models`、`mquake.checkpoint_root` 和历史实验目录。模板使用仓库相对路径；从仓库根目录运行，或改为自己的绝对路径。
3. 按需要安装实验依赖：`python -m pip install -e ".[experiments]"`。

查看通用入口：

```bash
rebind --help
python scripts/run_final_plan.py --help
```

最终实验入口依赖调用方准备的拆分、候选、训练清单及检查点，并在锁定评测时验证选择结果。此代码仓库不包含这些历史实验状态，不能在空目录直接恢复既有实验。原生基线适配器也需要对应上游代码。

## 导出说明

核心算子沿用已运行的项目实现。导出时移除了服务器专用路径，补充了安装元数据，并让缺少外部依赖的测试明确跳过。没有将历史 Git 仓库或其数据文件带入此仓库。

这是研究实现，代码可运行不等于已经证明方法优于基线；本仓库不附带性能或机制验证结论。

## 第一阶段：公共前端执行修复（v2）

第一阶段引入了 `action_bound_literal_provenance_v2`；下述动作一致性机制由当前 v3 版本保留。JSON、BP 和 ReBind 共享这套前端：

- 每次检索动作携带 `slot_id`、`input_values` 和调度时的 `evidence_version`。检索后优先按该动作的真实输入抽取，再处理保留绑定中的其他不同输入组合。
- 抽取任务采用当前可见证据版本。相同任务和相同证据只尝试一次；新证据可触发新的抽取。每个关系与输入组合最多发起一次定向检索，总检索预算仍由原配置限制。
- 轨迹分别记录 `retrieval_attempt`、`extraction_tasks.completed`、`produced_candidates`；成功返回空候选不作为抽取失败。多分支抽取可能增加生成调用，需要在真实实验中单独计量。
- 检索候选必须在其引用片段中出现，输入需出现在引用或标题上下文中。`literal_provenance` 仅表示通过字面核验，`relation_checked=False`，不会自动获得语义 `verified`。关系、方向和作用域保存在 `relation_claim` 中，但本阶段没有引入语义关系验证器。
- 未改变神经网络方程、参数形状或训练损失。新前端产生的候选与轨迹不同，历史锁定评测结果不能作为新版本结果复用。

运行定向回归与无标签轨迹审计：

```bash
python -m pytest tests/test_frontend_execution.py -q
python scripts/audit_frontend_execution.py /path/to/full_trace.json
```

轨迹审计只检查执行一致性；无法从未带标签的轨迹推断正确候选覆盖率、真实纠正事件或 QA 增益。后续解码复核、来源聚合、门控和重新训练属于独立阶段。

## 调试报告修复（v3）

当前前端版本：`candidate_identity_relation_constraints_v3`。

- **统一状态验证**：`assignments.py` 为 JSON 和神经路径提供相同的来源输入检查与 graph constraint 检查。内部用 `candidate_assignments` 保存候选 ID；`assignments` 仅保留显示文本。不能唯一对应候选的旧式名称输出保持 UNKNOWN。同名的父候选通过 `input_candidate_ids` 区分，候选 ID 也不会自动被视为已证实的现实实体 ID。
- **关系准入**：`configs/final.yaml` 默认 `relation_policy: strict`。共享模型核验引用中的输入、输出、关系、方向和作用域；只有得到正向关系支持的检索候选才能进入严格模式的绑定。背景假设与仅共同出现的候选不进入严格绑定。核验模型可能出错，因此 `verified` 仍不自动升级为真。
- **兼容诊断模式**：显式使用 `relation_policy: literal` 可复现仅按字面来源准入的策略；旧调用若省略该设置仍采用 literal。严格模式改变了可用候选与检索行为，两种策略的分数必须分别报告。
- **编码与空证据**：E5 token 编码保留全部 batch 并统一 padding；空输入返回具有正确维度的空结果。空检索使用未关联任何文档的数值 padding，输出里不伪造来源。单候选的批处理三角更新也与单样本实现一致。
- **环境适配**：通过 `devices.retriever/generator/scorer/train` 或对应 `REBIND_*_DEVICE` 选择设备；auto 在可用时选择 CUDA 0，否则选择 CPU。输入维度由编码器、特征或检查点确定；编码器与旧检查点不匹配时明确报错，不静默加载。
- **生成身份与结束原因**：支持单文件及分片 safetensors/PyTorch 权重，并散列实际权重文件。优先保留服务返回的 finish_reason，仅在缺失时使用 token 数回退判断；升级生成缓存版本，避免沿用旧结束原因记录。

### 固定检查点的开发集重跑

该入口不训练、不重新选择检查点，仅在已使用过的开发数据上验证修复：

```bash
python scripts/run_debug_validation.py --config /path/to/runtime.yaml --stage smoke --partition build --seeds 17
python scripts/run_debug_validation.py --config /path/to/runtime.yaml --stage dev --partition build --seeds 17 29 43
python scripts/run_debug_validation.py --config /path/to/runtime.yaml --stage dev --partition check --seeds 17 29 43
```

调用方需准备 `manifests/split_manifest.json`、`manifests/frozen_checkpoints.json`（含 selected_checkpoints）、公开题目、编辑记忆和索引、离线评分标签及官方评分代码。问题和检查点选择在推理前锁定，标签仅在全部推理完成后用于评分。阶段文件身份不一致时必须使用新运行目录。

调试报告声称的跨行 SyntaxError 在审阅对应的 `1d71a01` 提交及本次检查中均未复现；实际通过语法编译及测试收集验证，未凭报告示意改写有效提示文本。
