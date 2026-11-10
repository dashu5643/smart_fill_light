# smart_fill_light · 智能补光单帧策略引擎

**给一张图片，算出该怎么打光**：输出一套打光策略，以及 SureShot 双灯板 216 颗灯珠（2 块 × 18 行 × 6 列，RGBW）每一颗的值。

一次输入对应一次输出，不做实时闭环。把结果发到灯板是可选的外围功能，算法本身不依赖硬件。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # 在 Python 3.13 上测试
.venv/Scripts/python scripts/infer.py --image photo.jpg
```

终端打印一行摘要，并在 `outputs/` 生成两个文件：

```
photo.jpg | 脸1 肤色不可判 | L3 紫光 S98% P-Heavy | dual 【75-75***190-100】 亮100% | 301ms
```

- `photo.json`：打光策略 + 灯板参数（格式见下文）
- `photo_report.jpg`：报告图（原图标注 | 灯板预览 | 决策依据），给人看的

常用参数：

| 参数 | 作用 |
|---|---|
| `--distance 30` | 灯板到人脸的距离（cm），越近补光饱和度越低；不填按 60cm |
| `--mode scene_priority` | 场景光优先（保留现场氛围）；默认 `skin_priority` 肤色优先 |
| `--object food_cn` | 画面里没有人脸时指定物品类别：`food_cn` `food_west` `dessert` `salad` `beauty` `jewelry` `plant` |
| `--dir 文件夹 --out 目录` | 处理一批图片 |
| `--no-frame` / `--no-report` | JSON 里不写逐珠数据 / 不出报告图 |

在代码里调用：

```python
from fill_light import FillLightPipeline, imread_rgb

with FillLightPipeline() as pipe:                    # 人脸模型只加载一次，可反复 run
    res = pipe.run(imread_rgb("photo.jpg"), distance_cm=60)

res.strategy        # 打光策略：场景结论、灯型、颜色、决策依据、警告、备选
res.light_params    # 灯板参数（精简）：灯型 + 两路颜色 + 亮度
res.frame.leds      # 灯板参数（逐珠）：leds["left"][行][列] = [R, G, B, W]，0-255
res.to_dict()       # 以上全部，可直接存成 JSON
```

## 输出

| 字段 | 内容 |
|---|---|
| `env_light` | 环境光分类：类别 L0-L6、色相、饱和度、污染等级、是否多色、曝光 |
| `perception` | 看到了什么：人脸数、肤色等级、光线方向、瑕疵、环境光的两路估计 |
| `strategy` | 怎么打光，以及为什么：`scenario` 场景结论、`reasons` 决策依据、`warnings`、`alternatives` 备选 |
| `light_params` | 灯板参数 · 精简版（见下表） |
| `frame` | 灯板参数 · 逐珠版：216 颗 RGBW、电流钳位系数、平均占空比 |

`light_params`：

| 字段 | 说明 |
|---|---|
| `light_type` | `mono` 单色（全板同色）/ `sandwich` 夹心 / `dual` 内外圈互补 |
| `main` | 靠屏幕的列的颜色 `{name, hue, sat, hex}`：夹心 = 面部主光（4 列），内外圈 = 内圈（2 列） |
| `contour` | 外侧的列的颜色：夹心 = 轮廓光（2 列），内外圈 = 外圈（4 列）；单色时与 `main` 相同 |
| `t1_format` | 内外圈时的简写 `【外圈H-S***内圈H-S】`，如 `【75-75***190-100】` |
| `brightness_pct` | 亮度 0-100 |
| `side_boost_pct` / `top_gradient_pct` / `eye_patch` | 侧光补偿、顶光渐变、眼周修饰 |

颜色统一用 HSV：`hue` 0-360°，`sat` 0-100%。

灯板坐标是**观察者视角**（站在灯前、面向灯板）：行 0 在最上；两块板的列都从左到右编号，所以左板列 5、右板列 0 靠屏幕。

## 算法流程

```
图片 → ① 感知 → ② 分析 → ③ 决策 → ④ 执行 → 结果
```

| 层 | 做什么 |
|---|---|
| ① 感知 `perception` | MediaPipe 478 点找人脸（最多 6 张），在额头、两颊、下巴、眼下采样肤色；统计背景的光源颜色、色相分布、明暗 |
| ② 分析 `analysis` | 环境光归类 L0-L6，污染等级按饱和度分轻 / 中 / 重（15 / 30 / 60）；肤色 ITA → L1-L6 → 浅 / 中 / 深；光线方向（逆 / 侧 / 顶）；瑕疵；距离 |
| ③ 决策 `strategy` | 按场景分流（见下表），再定亮度并做侧光、顶光、眼周修饰 |
| ④ 执行 `hardware` | 灯型 → 每一列用哪个颜色；RGB → RGBW（公共白交给 W 通道）；整机电流钳位 |

决策分流：

| 场景 | 策略 |
|---|---|
| 画面太暗 | 暖白满亮，只提亮 |
| 没有人脸 | 物品模式：按指定类别打光；有色光污染时互补矫正；否则模仿环境光 |
| 多张人脸 | 两人同肤色按单人处理；否则用对所有肤色都安全的颜色 |
| 单人 · 肤色优先 · 有色光污染 | 内外圈互补色。饱和度 = T1 实测基准 × 污染系数 × 距离系数 × 多色系数 |
| 单人 · 肤色优先 · 无污染 | 肤色 × 环境基调 → 单色或夹心，再做底调修正和禁忌色替换 |
| 单人 · 场景光优先 | 轮廓光贴近场景色，面部主光降饱和并向肤色靠拢 |

强色光下肤色认不准，此时标记为"不可判"，按中等肤色的参数补光，并在 `warnings` 里说明。

## 工程结构

```
configs/
  strategy.yaml        算法参数表：所有阈值、色值、规则都在这里，代码里只有引擎
  bench.yaml           灯板下发的连接参数（可选功能）
models/                人脸关键点模型
src/fill_light/
  pipeline.py          入口：FillLightPipeline.run()，把四层串起来
  result.py            结果对象 Result 和输出的 JSON 结构
  perception/          ① 感知：face 人脸检测与采样 · environment 背景观测
  analysis/            ② 分析：ambient 环境光 · skin 肤色 · direction 光线方向 · blemish 瑕疵 · distance 距离
  strategy/            ③ 决策：engine 分流 · skin_priority · scene_priority · multi_face · object_mode
                               · dark_scene · complement 互补色 · modifiers 修饰 · palette 色板
  hardware/            ④ 执行：layout 灯板几何 · rgbw 通道换算 · renderer 策略 → 逐珠帧
  report/              报告图（可选，只给人看）
  bench/               灯板下发（可选）：mapping 坐标重排 · payload 限流 · link 连接 · records 验收记录
  color.py · images.py · config.py · labels.py · cli.py     通用工具
scripts/               命令行入口：infer 单帧推理 · to_bench 灯板工具 · live 连续运行 · eval_t1 评估
tests/                 测试
```

每一层只依赖它前面的层，层与层之间用各自 `types.py` 里的数据类传递：
`Observation → Analysis → Strategy → LedFrame`。要改某个场景的规则，只动 `strategy/` 里对应的那个文件。

## 调参

改 `configs/strategy.yaml`，不用改代码。常用的几节：

| 配置节 | 内容 |
|---|---|
| `cast` | 环境光的色相分类边界、污染等级阈值、多色模式 |
| `complement_table` | T1 互补色参数表（环境光 × 肤色组 → 外圈 / 内圈） |
| `distance` / `pollution_sat_scale` | 距离衰减、污染系数 |
| `base_light` / `palette` | 无污染时的肤色光、命名色板 |
| `multi_face` / `object_mode` | 多人安全色、物品类别 |
| `brightness` / `direction` / `blemish` | 亮度、光线方向补偿、瑕疵修饰 |
| `hardware` | 灯板行列数、内侧列数、电流钳位上限 |

## 下发到灯板（可选）

经 ss2-light-bench 台架工具（另一个仓库，默认放在本项目的同级目录）发到实物灯板，连接参数在 `configs/bench.yaml`。

```bash
python scripts/to_bench.py --caps                      # 能不能连上
python scripts/to_bench.py --identify                  # 方位测试：左板红、右板蓝、靠屏幕一列白、顶行绿
python scripts/infer.py --image photo.jpg --send       # 算完直接下发
python scripts/to_bench.py --off                       # 关灯

python scripts/live.py                                 # 连续运行：输入图片路径，回车即下发
python scripts/live.py --watch inbox                   # 连续运行：往文件夹里放新图片就处理
python scripts/live.py --camera 0 --show               # 连续运行：摄像头，决策变了才下发
```

- 不接板子：在 ss2-light-bench 目录运行 `py ss2light.py fake` 起一个假板端，这边的命令加 `--fake`；只想看下发内容加 `--dry-run`。
- 方位测试与预期不一致时，改 `bench.yaml` 的 `swap_boards` / `mirror_cols` / `flip_rows`。
- 台架整灯电流上限约 1 A，超过时整体等比调暗，颜色比例不变。

## 测试与评估

```bash
python -m unittest discover tests        # 21 项，不需要照片和灯板
python scripts/eval_t1.py                # 在 T1 实拍集上评估（数据不入库，需放到 testset/t1/）
```

T1 实拍集（《色彩矫正测试》162 张）上的结果：

| 指标 | 结果 |
|---|---|
| 无补光 · 环境光色系识别 | 42 / 44（95%） |
| 无补光 · 肤色组 | 41 张中 32 张标为"不可判"，判对 4，判错 5 |
| 闪光灯 · 环境光色系识别 | 23 / 36（64%） |
| 补光后污染饱和度平均下降 | 低饱和方案 29.8，高饱和方案 32.6（百分点） |

## 已知局限

1. **距离需要调用方给**。按脸宽估出来的是镜头距离，和灯距不是一回事，所以默认不估计，不传 `--distance` 时按 60cm。
2. **闪光灯加色光时肤色组识别弱**（36 张判错 19 张），白人模特头常被判成中 / 深肤色。
3. **黄光、青光的互补参数是理论值**，没有实测过。
4. 《环境光分类标准》§5.3 的饱和度上限与 T1 实测表冲突，默认不启用（`skin_sat_caps.enforce`）；深肤色禁冷色与 T1 粉光的青绿互补也冲突，以实测为准。
5. 物品类别要调用方指定；面中泛红只提示，不联动灯光。

## 依据文档

《智能补光系统场景识别算法需求文档 V2.0》《分区智能补光功能需求文档》《环境光分类标准》《环境光污染_互补色匹配参数表》（T1）《多人场景补光方案》《色彩矫正测试.xlsx》。`configs/strategy.yaml` 里各节的注释标了数值出处。
