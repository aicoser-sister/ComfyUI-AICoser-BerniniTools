# ComfyUI-AICoser-BerniniTools

Bernini 条件化节点，支持 12 种任务模式，集成正面/负面提示词编码、VAE 编码、context_latent 构造。

## 功能

- 12 种任务模式：视频编辑、参考图生成视频、文本生成视频/图片、图片编辑等
- 正面/负面提示词集成，负面词为空时自动使用默认中文负面词
- 参考图连线（image0~image7），提示词中用 `image0`、`image1` 引用
- 源视频/参考视频 VAE 编码
- context_latent 构造并附加到 conditioning
- 输出正面条件、负面条件、latent

## 安装

将本目录放入 ComfyUI 的 `custom_nodes` 文件夹下，重启 ComfyUI。

## 使用

1. `CLIPLoader` 加载 T5 文本编码器（选 wan type）
2. `VAELoader` 加载 Wan VAE
3. 连入 `BerniniTools` 节点
4. 选择任务模式，填写提示词
5. 输出连接到 `SamplerCustom`

## 任务模式

| 模式 | 说明 | 需要连接 |
|---|---|---|
| v2v | 视频编辑 | 源视频 |
| rv2v | 参考图+视频编辑 | 源视频 + image0 |
| r2v | 参考图生成视频 | image0 |
| t2v | 文本生成视频 | 无 |
| t2i | 文本生成图片 | 无 |
| r2i | 参考图生成图片 | image0 |
| i2i | 图片编辑 | source_video (单帧) |
| i2v | 图片转视频 | image0 |
| mv2v | 动作编辑 | 源视频 |
| vi2v | 内容传播 | 源视频 + image0 |
| ads2v | 广告插入 | 源视频 + image0 |
| vrc2v | 视频重定向 | 源视频 + image0 |

## 视频教程

[一个节点搞定多图参考生视频，体验直逼Seedance2.0！自制Bernini Tools节点](https://www.bilibili.com/video/BV1MVTd64EsQ/)
