# VDS90 → MUB478

当前发布版保留方向文件、划分、汇总指标和对比报告，省略逐题回答、
隐藏状态、逐题校准记录和 `crossfit.jsonl`。下文的采集、续跑与重建比较
功能适用于新运行生成的记录；保留的历史目录不能用于续跑完整实验。
复现实验请传入新的 `--output`，重算 MUB30/VDS90 对比还需要新生成的
MUB30 逐题记录并通过 `--reference` 指向它。已有 `comparison.json/csv/md`
可直接阅读。

在 VDS 固定 90 题上重新提取方向，使用与现有 MUB30 提取完全相同的提示词、选项文本评分、三折选层与符号校准；冻结后直接用于 MUB 的 478 题留出测试。三种模型分别提取自己的方向。

```text
src/vds_mub/
├── run.py
├── data.py
├── report.py
├── qwen2.5_vl_7b/config.json
├── llava_next_7b/config.json
├── internvl3_8b/config.json
└── tests/
```

模型与配置沿用已有 MUB 实验：Qwen2.5-VL-7B-Instruct、LLaVA-NeXT v1.6 Mistral-7B、InternVL3-8B。每个子目录保留相应环境的 requirements.txt。InternVL3 与另外两组的 Transformers 版本要求不同，使用各自现有环境运行。原有 `src/mub/` 和 `src/steering/` 是本入口复用的实现，服务器需要完整项目目录。

VDS 源集明确为 `data/splits/binding_subspace_dev90.json`，三类各 30 题。保留原图、原问题内容、原选项、正确答案和 D+/D− 错误目标。统一调用各模型现有的 `mub.<model>.protocol.training_messages()`，第二轮使用 `I think the correct answer is: {option text}`，初答为该模型真实的无干预选项文本，两臂共享该初答。

提取使用所有 90 个配对状态，拟合一个 pooled 方向，沿用 MUB30 的拟合与选层准则；三折只在源集内部进行。符号由源集的 GOLD-minus-target 分数变化决定。MUB 测试题不用于拟合、选层、定号或挑强度。测试沿用 MUB 官方前缀、误导指令、生成参数和答案解析，独立检查第一轮能力。三档范数强度仍为 10%/20%/30%；保留两种符号的完整记录，主报告只列源集选定符号。

从项目根目录执行，默认 `--stage all` 自动完成采集、拟合、校准、测试与对比：

```bash
python -u src/vds_mub/run.py --model qwen
python -u src/vds_mub/run.py --model llava
python -u src/vds_mub/run.py --model internvl
```

无权重的数据检查：

```bash
python src/vds_mub/run.py --model qwen --stage validate
python src/vds_mub/run.py --model llava --stage validate
python src/vds_mub/run.py --model internvl --stage validate
```

指定权重路径或分阶段运行：

```bash
python -u src/vds_mub/run.py --model qwen --model-path /path/to/Qwen2.5-VL-7B-Instruct
python -u src/vds_mub/run.py --model qwen --stage collect
python src/vds_mub/run.py --model qwen --stage fit
python -u src/vds_mub/run.py --model qwen --stage calibrate
python -u src/vds_mub/run.py --model qwen --stage evaluate
```

重复相同命令可利用已有完整逐题记录续跑。输出默认为 `results/vds_mub/<模型目录>/`，可通过 `--output` 修改。源码、配置、数据和图片身份写入 manifest，不同身份会拒绝续写同一目录。

输出包括 `preflight.json`、转换后的 `source_vds90.json`、`v1/axis.npy`、三折与符号校准记录，以及 `v2/records/` 和 `v2/summary.json`。第一轮或第二轮无干预答案错误的样本仍保留，主要谄媚分母固定为无干预初答正确的测试题。

默认与 `results/mub/<qwen|llava|internvl>/` 中已有的 MUB30 结果比较，主表严格使用同一 MUB478，不混入其 30 道训练题。其他对照路径可用 `--reference /path/to/mub30_run` 指定，目录应包含 `v1/` 和 `v2/`。两组必须使用相同权重路径、配置、协议和测试题。

比较前逐题检查 baseline 初答与受误导后生成的文本及答案；出现不一致会标为 `baseline_mismatch`，不把不同基线悄悄视为同一次实验。对照缺失则标为 `reference_unavailable`，VDS90 的测试结果仍保存在 v2 下。完整比较输出为 `comparison.json`、`comparison.csv`、`comparison.md`，包括错误目标采纳率、任何错误率、初答准确率、原本正确题保留率及逐题修复/伤害。

独立重建比较报告：

```bash
python src/vds_mub/run.py --model qwen --stage compare
```

VDS90 与 MUB30 的样本数量不同，这组实验用于检验跨数据集迁移，不能单独归因于数据来源优劣。这里的方向在各模型自身权重上重新提取，不跨模型共享向量。该目录与 MM-SY attention baseline 分别运行。

CPU 测试：

```bash
python -m unittest discover -s src/vds_mub/tests -v
```

测试中的合成引擎仅检查数据、提示词、源集选层校准、冻结评估、续跑与 baseline 对齐，不产生实际模型结果。真实权重和环境检查沿用各模型原有的 MUB smoke 工具。
