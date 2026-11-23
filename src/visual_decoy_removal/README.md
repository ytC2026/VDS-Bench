# 视觉诱饵移除附加实验

本实验按项目统一结构组织：代码位于 `src/visual_decoy_removal/`，数据位于 `data/visual_decoy_removal/`，结果位于 `results/visual_decoy_removal/`，专用环境配置位于 `configs/visual_decoy_removal/`。以下路径均相对于项目根目录。

本实验在同一批 100 道题上比较原图和编辑图。编辑目标是移除 D+ 错误建议对应的视觉诱饵，同时保留正确答案证据。问题、选项、正确答案及两种反馈指定的错误答案在编辑前后保持一致。每题有原图 D+、原图 D−、编辑图 D+、编辑图 D− 四个条件。

## 文件组织

| 路径 | 内容 |
|---|---|
| `data/visual_decoy_removal/dplus.jsonl` | 100 道题的 D+ 反馈目标及证据标注 |
| `data/visual_decoy_removal/dminus.jsonl` | 相同 100 道题的 D− 反馈目标及证据标注 |
| `data/visual_decoy_removal/images/original/` | 100 张原图，以 `pair_id` 命名 |
| `data/visual_decoy_removal/images/edited/` | 100 张编辑图，以相同 `pair_id` 命名 |
| `data/visual_decoy_removal/筛选标准.txt` | 题目、诱饵和编辑质量的筛选标准 |
| `src/visual_decoy_removal/run.py` | 四条件 API 评测入口，支持本地数据检查 |
| `src/visual_decoy_removal/analyze.py` | 按固定配对样本汇总结果 |
| `src/visual_decoy_removal/bootstrap_gap.py` | 按题配对 bootstrap 的差距及差距缩小量分析 |
| `results/visual_decoy_removal/` | 四个模型的逐题记录、运行清单和统计结果 |

JSONL 中的 `image_path`、`original_image_path`、`edited_image_path` 均相对于项目根目录下的 `data/visual_decoy_removal/`。`image_path` 默认指向原图，评测入口按条件选择图片。D+ 和 D− 通过 `pair_id` 对齐，文件行序相同。

100 题包含指代或属性重绑定 25 题、空间范围偏移 37 题、对象或部件替换 38 题。该分布为现有编辑实验的实际组成。

## 已有结果

以下差距为 D+ 谄媚率减 D− 谄媚率，单位为百分点。每个模型使用其四个条件下均初答正确且解析有效的共同样本计算编辑前后差距。各模型的共同样本不同。

| 模型 | 请求题数 | 共同有效题数 | 原图差距 | 编辑图差距 | 差距缩小量 |
|---|---:|---:|---:|---:|---:|
| Qwen3.7-Plus | 100 | 98 | 43.88 | 7.14 | 36.73 |
| Kimi-K3 | 100 | 94 | 11.70 | 1.06 | 10.64 |
| MiniMax-M3 | 100 | 98 | 21.43 | 2.04 | 19.39 |
| MiMo-V2.6-Pro | 100 | 97 | 12.37 | -1.03 | 13.40 |

数值来自各模型 `analysis.json` 的 `all_four_matched`。Kimi 编辑图 D+、D− 分别有 1、2 条解析失败，已按共同有效样本规则处理。报告中各条件独立分母的谄媚率与本表共同分母的统计口径不同。图片编辑的局部性及正确答案证据保留情况需要结合原图和编辑图审阅。

## 使用入口

在项目根目录执行本地检查。

```bash
python -B src/visual_decoy_removal/run.py --check
```

查看 API 评测参数。

```bash
python -B src/visual_decoy_removal/run.py --help
```

重建某个已有运行的统计报告。

```bash
python -B src/visual_decoy_removal/analyze.py --run results/visual_decoy_removal/qwen3.7-plus
```

运行入口基于自身文件位置定位项目根目录，默认数据和结果路径不依赖当前工作目录。`--data-dir` 接受绝对路径或相对于项目根目录的路径。`--run-name` 指定 `results/visual_decoy_removal/` 下的运行名称。

默认优先读取 `configs/visual_decoy_removal/.env`，不存在时回退到根目录 `.env`；也可用 `--env-file` 显式指定。原专用配置原样迁入，示例见同目录 `.env.example`。实验内部的 `api_eval/` 保留原图/编辑图切换及并发实现，通过 `visual_decoy_removal.api_eval` 导入，不覆盖项目通用的 `src/api_eval/`。

安装项目后也可使用 `vds-decoy-removal --check`。从项目根目录执行 bootstrap：

```bash
python -B src/visual_decoy_removal/bootstrap_gap.py --run results/visual_decoy_removal/qwen3.7-plus
```

本实验先于 2026-11-07 从 `src/try/edit/` 整体迁入 `experiments/visual_decoy_removal/`，再于 2026-11-08 归并到项目统一目录。此次迁移在修改代码前逐文件校验了代码、数据、结果和专用环境配置的 SHA-256。数据与历史结果保持原样；已有运行清单和执行历史可能包含旧路径及旧源码哈希，它们是历史记录，不作为当前运行命令。新评测应使用新的 `--run-name`，不能在源码已变更后续跑旧清单。已有结果可直接用上面的分析命令重新汇总。
