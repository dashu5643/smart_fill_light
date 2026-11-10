# 智能补光 · 单帧策略引擎（smart_fill_light）

输入一张图片，输出一套补光策略，并渲染成 SureShot 双灯板（2 × 18 行 × 6 列 × RGBW）的逐珠帧。
不做实时、不做闭环；一次输入 → 一次输出。

依据文档：
- 《智能补光系统场景识别算法需求文档 V2.0》—— 7 步流水线、L1-L6 肤色、10 类环境基调、肤色×环境矩阵、物品/多人分支
- 《分区智能补光功能需求文档》—— 肤色优先 + 互补色矫正；多色分区控光无效 → 单色灯 / 夹心灯
- 《智能补光——场景应用》—— 肤色优先 / 场景光优先、多人安全色、物品打光
- 《环境光分类标准》—— L0-L6 七类环境光、P-Light/Mid/Heavy 污染等级、多色模式、§7 输出字段
- 《环境光污染_互补色匹配参数表》（T1）—— 【外圈H-S***内圈H-S】内外圈互补、距离衰减、污染系数
- 《多人场景补光方案》—— 同肤色 2 人走单人策略，否则按环境光选安全交集色
- 《色彩矫正测试.xlsx》—— 162 张实拍（4 色光 × 3 肤色 × 2 设备 × 2 距离 × 闪光/无/高饱和/低饱和），已提取到 `testset/t1/`
- SureShot 架构讲解 —— 灯板 216 颗 × RGBW、电流钳位

## 快速使用

```bash
cd F:/smartlight/smart_fill_light
.venv/Scripts/python scripts/infer.py --image samples/zone_image7.png            # 肤色优先（默认）
.venv/Scripts/python scripts/infer.py --image xxx.jpg --distance 30              # 指定灯到人脸距离(cm)
.venv/Scripts/python scripts/infer.py --image xxx.jpg --mode scene_priority      # 场景光优先
.venv/Scripts/python scripts/infer.py --image food.jpg --object food_cn          # 物品模式
.venv/Scripts/python scripts/infer.py --dir samples --out outputs/batch --no-frame
.venv/Scripts/python scripts/eval_t1.py                                          # 在 T1 实拍集上评估
```

每张图输出 `<名>.json`（env_light + 感知 + 策略 + light_params + 216 颗 RGBW 帧）和 `<名>_report.jpg`。

```python
from fill_light.pipeline import FillLightPipeline, imread_rgb
pipe = FillLightPipeline()
res = pipe.run(imread_rgb("a.jpg"), mode="skin_priority", distance_cm=60)
res["env_light"]      # 《环境光分类标准》§7 字段
res["light_params"]   # 精简 HSV 参数（含 t1_format）
res["frame"]["leds"]  # {"left": 18×6×4, "right": 18×6×4}
```

## 下发到灯板（ss2-light-bench 台架）

链路：`取景图 → pipeline.run → frame.leds(观察者视角 2×18×6 RGBW) → bench.py 重排成 216 颗编号顺序 → ss2light.Light.apply → USB-C 网卡 → V1.5 主板 → 灯板`

```bash
.venv/Scripts/python scripts/to_bench.py --caps                     # 连通性
.venv/Scripts/python scripts/to_bench.py --identify                 # 方位：左红/右蓝/靠屏幕一列白/顶行绿
.venv/Scripts/python scripts/to_bench.py --image 取景.jpg --distance 60
.venv/Scripts/python scripts/to_bench.py --result outputs/test/xxx.json
.venv/Scripts/python scripts/to_bench.py --dir testset/t1 --hold 3  # 轮播
.venv/Scripts/python scripts/to_bench.py --off
.venv/Scripts/python scripts/infer.py --image 取景.jpg --send       # 推理 + 出报告 + 下发，一条命令
```
注意：`infer.py` 不加 `--send` 只推理、不碰灯板。

**连续运行**（模型只加载一次，Ctrl+C 退出）：
```bash
.venv/Scripts/python scripts/live.py                              # 交互：输入/拖入图片路径回车即下发，q 退出
.venv/Scripts/python scripts/live.py --watch inbox                # 往 inbox 文件夹放新图就自动处理
.venv/Scripts/python scripts/live.py --camera 0 --interval 1 --show   # 摄像头每秒决策一次，决策变了才下发
```
摄像头模式下同一新决策需连续出现 `--stable`（默认 2）次才切换，色相差 <10°、饱和度差 <5%、亮度差 <5% 视为没变，避免灯板闪。
- 不接板子：在 `ss2-light-bench` 目录跑 `py ss2light.py fake`，这边加 `--fake`；只看下发内容加 `--dry-run`。
- 每次下发自动存验收记录到 `outputs/bench/records/<时间>_<名>/`（取景图、算法结果、下发 JSON），灯光照片手动补进去（09_algo C-6）。
- 配置在 `strategy.yaml → bench`：板子地址、渐变时长、电流上限、方位校正 `swap_boards / mirror_cols / flip_rows`。
- 台架整灯电流上限 1 A：算法满亮帧约 3.2 A，会整体等比缩到 ~30%（颜色比例不变，只是暗）。

Python 调用：
```python
from fill_light.bench import BenchLink
link = BenchLink(pipe.cfg)              # 默认 192.168.7.1
link.send_result(res)                   # res = pipe.run(...)
```

## 算法流程

```
图片
 └─ ① 感知 perception.py
     ├─ MediaPipe 478 点（最多 6 张脸），6 区域采样取 Lab 中位数
     └─ 背景：Shades-of-Gray 光源、霓虹色相峰、全画面平均 V/S（曝光判定）
 └─ ② 分析 analysis.py
     ├─ 曝光：暗光 / 过曝 → 不判色光污染
     ├─ 环境光：人脸色卡反推 + 背景估计融合 → L0 中性 / L1 蓝 / L2 绿 / L3 紫 / L4 暖粉 / L5 黄 / L6 青
     │     S% → P-Light(15-30) / P-Mid(30-60) / P-Heavy(≥60)；≥2 个色相峰 → 多色模式
     ├─ 肤色：去偏色 → ITA → L1-L6 → 浅/中/深；强色光下（S≥60 / 通道缺失 / 过暗）标记"不可判"
     ├─ 光线方向：逆/侧/顶（肤色不可判时跳过侧/顶判断）
     └─ 瑕疵（肤色不可判时跳过）
 └─ ③ 决策 strategy.py
     ├─ 无人脸 → 物品模式；实测过的色光污染下做内外圈互补
     ├─ 多人   → 同肤色 2 人走单人策略；否则按环境光选安全交集色（紫→暖黄绿 / 粉→淡橙 / 蓝→淡黄 / 绿→淡粉）
     ├─ 暗光   → 暖白满亮
     ├─ 单人·有污染 → T1 表（环境光 × 肤色组）内外圈互补 dual
     │     S = T1 基准 × 污染系数(1.0/0.85/0.7) × 距离系数(60cm 1.0 / 30cm 0.53 / 20cm 0.4) × 多色系数
     │     并给出低饱和 / 高饱和备选
     ├─ 单人·无污染 → 肤色×环境基调矩阵（单色/夹心）+ 底调修正 + 禁忌
     └─ 修饰：污染时满亮；侧光暗侧板 +25%（主亮度预留余量）；顶光渐变；眼周修饰（dual 不做）
 └─ ④ 执行 hardware.py
     ├─ dual：靠屏幕 2 列 = 内圈（浓），其余 4 列 = 外圈；sandwich：内 4 列主光 + 外 2 列轮廓
     └─ RGB → RGBW（公共白走 W）→ 电流钳位
```

## 输出协议

`env_light`：env_light_category(L0-L6) / env_light_name / hue / saturation / brightness / pollution_level /
multi_color_mode / secondary_category / face_detected / skin_tone_type(Fair/Medium/Deep/Unknown) / exposure / confidence / timestamp

`light_params`：

| 字段 | 说明 |
|---|---|
| light_type | `mono` 单色 / `sandwich` 夹心 / `dual` 内外圈互补 |
| main / contour | dual 时 main=外圈、contour=内圈：`{name, hue, sat, hex}` |
| t1_format | 例 `【75-75***190-100】` |
| brightness_pct / side_boost_pct / top_gradient_pct / eye_patch | 亮度与修饰 |

逐珠帧 `frame.leds.left/right[row][col] = [R,G,B,W]`；左板 col0=外侧→col5=内侧，右板相反。

## 调参（configs/strategy.yaml）

- `distance`：`estimate_from_face`（默认 false，未传 `--distance` 时按 60cm）、`attenuation`
- `cast.hue_classes` / `levels` / `multi_color`：环境光分类（已用 T1 实拍校准边界）
- `complement_table` / `complement_variants` / `pollution_sat_scale`：T1 互补参数
- `multi_face`：多人安全色；`skin.reliability`：肤色不可判阈值
- `direction.skip_side_top_when_skin_unreliable`：强色光下不判侧/顶光

## T1 实拍评估（scripts/eval_t1.py）

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 无补光·环境光色系正确率（44 张） | 82%（蓝常判成紫） | **95%** |
| 无补光·肤色组判错 | 27 | **5**（32 张标"不可判"，按中肤参数） |
| 闪光灯·色系正确率（36 张） | — | 64% |
| T1 补光后污染 S% 平均下降 | — | 低饱和 29.8 / 高饱和 32.6 |

## 已知局限 / 待确认

1. **灯距**：脸宽估的是镜头距离，T1 中与灯距不同步 → 默认不估计，需调用方传 `--distance`。
2. **§5.3 饱和度上限**与 T1 实测表冲突（T1 白人粉光用到 S100），默认不启用（`skin_sat_caps.enforce`）。
3. **深肤禁冷色** vs T1 粉光给青绿：以实测为准。
4. **黄光 / 青光**互补参数为理论值，未实测。
5. **闪光灯 + 色光**下肤色组识别弱（19/36 判错），白人模特常判成中/深。
6. 物品类别需外部指定 `--object`；面中泛红仅提示不联动。
