[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# qdrift

对仿射（affine）INT8 量化运算（即 `QuantizeLinear`/`DequantizeLinear` 风格的
scale + zero-point 数学，被 ONNX Runtime、TensorFlow Lite、ARM Compute
Library 等使用）与一个精确有理数参照实现进行对比检查。`qdrift` 并不运行真实
的推理框架或硬件内核——它重建了两类已被报告的偏差问题的**问题形状**，并提供
一个可复用的精确参照，供其他工具或手写内核对照检查。

## 检查内容

**舍入边界行为**（`check-round`）：候选量化内核在处理"恰好落在 `.5` 边界"
的数值（除以 `scale` 后商恰好为 .5）时，是否遵循 ONNX 规范的"就近取偶"
（round-half-to-even），还是采用了其他规则。重建了
[microsoft/onnxruntime#18576](https://github.com/microsoft/onnxruntime/issues/18576)
的问题形状——该issue报告 `QuantizeLinear` 内核在处理边界值时始终向下舍入，
而非按规范就近取偶。

**跨scale逐元素加法**（`check-add`）：将两个具有**不同** scale/zero-point
的 INT8 张量相加，并重新量化到第三个 scale 时，结果是否数学正确。重建了
[openvinotoolkit/openvino#34673](https://github.com/openvinotoolkit/openvino/issues/34673)
的问题形状——这是一个开放的、已被独立复现的issue，报告了在 Apple M4 硬件上
INT8 残差加法（residual add）结果损坏的问题。

**明确说明的局限性：** `qdrift` 无法访问 ONNX Runtime 真实的
`QuantizeLinear` 内核，也无法访问 ARM Compute Library 真实的
NEQuantizationLayer/Add 内核，更无法在 M4 专属的 ACL 代码路径上运行。它提供
了两个示例性的候选实现（一个纯 float64 内核和一个 16 位定点累加器内核），
按照真实运行时内核常见的实现方式构建，并与精确参照对比——这是针对**这一类**
bug 的可复用正确性检验工具，而不是对上述两个issue具体根因的确认复现。如果
你在维护真实的量化内核，可以将 `qdrift.reference` 作为参照，替换成你自己的
内核调用。

## 安装

需要 Python 3.9+，无运行时依赖（仅使用标准库的 `fractions` 和 `decimal`）。

```bash
git clone https://github.com/zhuhroscar-tech/qdrift.git
cd qdrift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

发行版 wheel 包见 [Releases](https://github.com/zhuhroscar-tech/qdrift/releases)；
安装前请核对随附的 `SHA256SUMS.txt`。

## 快速开始

```bash
qdrift list-fixtures
qdrift check-round                                   # 默认：half_even（符合规范）
qdrift check-round --round-mode half_down_bug         # 故意有bug的模式，用于对比
qdrift check-add                                      # 默认：float64 内核
qdrift check-add --kernel int32_fixedpoint            # 定点累加器内核
qdrift check-round --json
```

所有探测值均与精确参照匹配时退出码为 `0`；发现任意不匹配（drift）时为 `1`。

## 参照实现的原理

所有"标准答案"运算均使用 Python 的 `fractions.Fraction`，因此参照本身不会
带有浮点舍入误差。`round_half_even` 在 `Fraction` 上精确计算 IEEE-754/ONNX
规范的就近取偶舍入——没有任何中间的浮点转换。`exact_requantize_add` 将两个
输入精确反量化为有理数，精确相加，再进行一次量化——这就是真实的浮点或定点
内核结果所要对照的、无歧义的"正确答案"。

## 设计说明

- 与本项目其他 CLI 一致：统一的状态图标系统、纯语义化的 ANSI 颜色、支持
  `NO_COLOR`/`--no-color`。
- 每一处发现都会报告具体输入值、精确结果码与候选结果码——绝不仅给出
  通过/失败的汇总。
- `near_identical_scales_sanity` 这个测试用例并非"理应总是通过"的对照组：
  内置的 `int32_fixedpoint` 内核在该用例上确实会发生偏差（在该用例自身的
  描述中已注明，并由一个回归测试固定住），因为即便两个 scale 非常接近，
  真实的和仍可能恰好落在定点重新缩放会跨越的舍入边界上。

## 许可证

MIT
