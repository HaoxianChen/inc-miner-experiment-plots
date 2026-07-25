# 连续多轮 Cutoff 更新实验结果

## 1. 实验配置

| 配置项 | 取值 |
|---|---|
| 数据集 | Adult |
| Cutoff 序列 | 0.9 → 0.8 → 0.7 → 0.6 → 0.5 |
| 更新轮数 | 4 |
| support | 0.000001 |
| confidence | 0.7 |
| 重复次数 | 1 |

## 2. 论文主图数据

论文主图注明 `initialization overhead not shown`，因此比较：

$$
T_{\mathrm{Inc,cumulative}}(r)
=
\sum_{i=1}^{r}T_{\mathrm{online},i}
$$

$$
T_{\mathrm{Batch,cumulative}}(r)
=
\sum_{i=1}^{r}T_{\mathrm{batch},i}
$$

| 轮次 | Cutoff 更新 | IncMiner 单轮 Online（ms） | BatchMiner 单轮（ms） | IncMiner Online 累计（ms） | BatchMiner 累计（ms） |
|---:|---|---:|---:|---:|---:|
| 1 | 0.9 → 0.8 | 1,526 | 493,013 | 1,526 | 493,013 |
| 2 | 0.8 → 0.7 | 1,705 | 493,543 | 3,231 | 986,556 |
| 3 | 0.7 → 0.6 | 1,470 | 493,348 | 4,701 | 1,479,904 |
| 4 | 0.6 → 0.5 | 2,339 | 477,623 | 7,040 | 1,957,527 |

### 绘图方式

- x 轴：更新轮次 `1、2、3、4`；
- y 轴：累计运行时间，单位 `ms`；
- IncMiner 曲线：`1,526、3,231、4,701、7,040`；
- BatchMiner 曲线：`493,013、986,556、1,479,904、1,957,527`。

第 4 轮的累计加速比：

$$
FinalOnlineSpeedup
=
\frac{1,957,527}{7,040}
=
278.06
$$

即：

```text
最终 Online 口径加速比 = 278.06×
```

## 3. Offline SampleStatistics 维护时间

Online 挖掘和 Offline SampleStatistics 更新分开统计。

| 轮次 | Offline 单轮（ms） | Offline 累计（ms） | Online + Offline 累计（ms） |
|---:|---:|---:|---:|
| 1 | 30 | 30 | 1,556 |
| 2 | 30 | 60 | 3,291 |
| 3 | 27 | 87 | 4,788 |
| 4 | 38 | 125 | 7,165 |

四轮 Offline 总时间：

```text
125 ms
```

当前结果下，Offline 时间相对于 Online 累计时间 `7,040 ms` 较小。

建议：

- 论文主图仍使用 Online 累计时间；
- Offline 时间作为补充结果单独报告；
- 如果后续决定采用 Online + Offline 口径，则使用表中第三列累计值。

需要注明：本次四轮的 `Φ_newRuleCount` 都为 `0`，没有触发新规则的 confidence 重算和 SampleNode 更新。因此，这组 Offline 时间主要是扫描与收集开销，不代表出现大量新规则时的 Offline 开销。

## 4. 初始化开销

| 指标 | 结果 |
|---|---:|
| 普通 Batch 初始化时间 `plain-init` | 465,283 ms |
| 增量就绪初始化时间 `inc-ready-init` | 578,429 ms |
| 初始化额外时间 | 113,146 ms |
| 初始化额外比例 | 24.32% |
| `preComputeSampleMaxConf` 时间 | 121,307 ms |
| 初始 SampleNode 数量 | 2,895 |

初始化额外时间：

$$
T_{\mathrm{init-overhead}}
=
T_{\mathrm{inc-ready-init}}
-
T_{\mathrm{plain-init}}
$$

$$
T_{\mathrm{init-overhead}}
=
578,429-465,283
=
113,146\ \mathrm{ms}
$$

初始化额外比例：

$$
InitializationOverheadPercent
=
\frac{
T_{\mathrm{inc-ready-init}}
-
T_{\mathrm{plain-init}}
}{
T_{\mathrm{plain-init}}
}
\times 100\%
$$

$$
InitializationOverheadPercent
=
\frac{113,146}{465,283}
\times100\%
=
24.32\%
$$

需要说明：`plain-init` 和 `inc-ready-init` 是两次独立运行，Batch 核心时间可能存在波动。因此，不能要求：

$$
T_{\mathrm{init-overhead}}
=
T_{\mathrm{preComputeSampleMaxConf}}
$$

本次 `preComputeSampleMaxConf=121,307 ms` 大于时间差 `113,146 ms`，属于两次基础 Batch 时间波动造成的正常现象。

## 5. 包含初始化成本的端到端摊销

端到端 IncMiner 累计时间：

$$
T_{\mathrm{Inc,total}}(r)
=
T_{\mathrm{inc-ready-init}}
+
\sum_{i=1}^{r}
\left(
T_{\mathrm{online},i}
+
T_{\mathrm{offline},i}
\right)
$$

端到端 BatchMiner 累计时间：

$$
T_{\mathrm{Batch,total}}(r)
=
T_{\mathrm{plain-init}}
+
\sum_{i=1}^{r}T_{\mathrm{batch},i}
$$

| 轮次 | IncMiner 端到端累计（ms） | BatchMiner 端到端累计（ms） |
|---:|---:|---:|
| 初始化 | 578,429 | 465,283 |
| 1 | 579,985 | 958,296 |
| 2 | 581,720 | 1,451,839 |
| 3 | 583,217 | 1,945,187 |
| 4 | 585,594 | 2,422,810 |

第 1 轮首次满足：

$$
T_{\mathrm{Inc,total}}(r)
<
T_{\mathrm{Batch,total}}(r)
$$

所以：

```text
端到端 break-even round = 1
```

第 4 轮端到端加速比：

$$
FinalEndToEndSpeedup
=
\frac{2,422,810}{585,594}
=
4.14
$$

即：

```text
最终端到端加速比 = 4.14×
```

## 6. 实验的核心结果

| 论文指标 | 当前结果 |
|---|---:|
| 更新轮数 | 4 |
| IncMiner Online 最终累计时间 | 7,040 ms |
| BatchMiner 最终累计时间 | 1,957,527 ms |
| 最终 Online 加速比 | 278.06× |
| Offline 最终累计时间 | 125 ms |
| `plain-init` | 465,283 ms |
| `inc-ready-init` | 578,429 ms |
| 初始化额外时间 | 113,146 ms |
| 初始化额外比例 | 24.32% |
| 端到端 break-even round | 1 |
| IncMiner 最终端到端累计时间 | 585,594 ms |
| BatchMiner 最终端到端累计时间 | 2,422,810 ms |
| 最终端到端加速比 | 4.14× |
