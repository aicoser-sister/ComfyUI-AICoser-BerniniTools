"""
BerniniTools - Bernini 条件化节点

- 12 种 task 模式 + system prompt
- 文本编码（正面/负面提示词集成）
- VAE 编码源视频/参考图
- context_latent 构造
- 输出 CONDITIONING + LATENT
"""

import logging
import torch

import comfy.model_management as mm
from comfy.utils import common_upscale

try:
    import node_helpers
except ImportError:
    node_helpers = None

log = logging.getLogger("BerniniTools")


# =========================================================================
# Task types: 下拉框显示中文标签
# =========================================================================

TASK_TYPES = [
    "v2v - 视频编辑",
    "rv2v - 参考图+视频编辑",
    "r2v - 参考图生成视频",
    "t2v - 文本生成视频",
    "t2i - 文本生成图片",
    "r2i - 参考图生成图片",
    "i2i - 图片编辑",
    "i2v - 图片转视频",
    "mv2v - 动作编辑",
    "vi2v - 内容传播",
    "ads2v - 广告插入",
    "vrc2v - 视频重定向",
]

# 从下拉框标签提取原始 task key
def _parse_task(label):
    return label.split(" - ")[0].strip() if " - " in label else label.strip()


SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": (
        "You are a helpful assistant for editing. "
        "You may need to adjust the subject's action or position."
    ),
    "mv2v": (
        "You are a helpful assistant for editing. "
        "You might need to adjust the video's style, lighting, colors, "
        "textures, and the subject's pose or action."
    ),
}

DEFAULT_NEG_PROMPT = (
    "\u8272\u8c83\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce"
    "\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c"
    "\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28"
    "\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0cJPEG\u538b\u7f29\u6b8b\u7559"
    "\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59"
    "\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8"
    "\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62"
    "\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c\u5f62\u6001\u7578\u5f62\u7684"
    "\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d"
    "\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c"
    "\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012"
    "\u7740\u8d70"
)


def _get_system_prompt(task_type):
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])


# =========================================================================
# Image helpers
# =========================================================================

def _resize_long_edge(image, max_size, stride=16):
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return common_upscale(
        image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
    ).movedim(1, -1)


# =========================================================================
# Node
# =========================================================================

class BerniniTools:
    """Bernini 条件化节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIPLoader 加载的 T5 文本编码器 (wan type)"}),
                "vae": ("VAE", {"tooltip": "VAELoader 加载的 Wan 2.1/2.2 VAE"}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4,
                    "tooltip": "输出帧数，Wan 网格: 4n+1 (81, 121, 145...)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "task_type": (TASK_TYPES, {"default": "v2v - 视频编辑",
                    "tooltip": "任务模式，决定 system prompt 和条件化行为"}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "正面提示词。参考图任务中用 image0, image1... 引用连接的参考图"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "负面提示词，留空则使用 Bernini 默认中文负面词"}),
                "use_default_neg": ("BOOLEAN", {"default": True,
                    "tooltip": "开启时，负面提示词为空则自动使用 Bernini 默认负面词"}),
            },
            "optional": {
                "source_video": ("IMAGE", {"tooltip": "源视频（v2v/rv2v/mv2v 用）"}),
                "image0": ("IMAGE", {"tooltip": "参考图 0，提示词中用 image0 引用"}),
                "image1": ("IMAGE", {"tooltip": "参考图 1，提示词中用 image1 引用"}),
                "image2": ("IMAGE", {"tooltip": "参考图 2，提示词中用 image2 引用"}),
                "image3": ("IMAGE", {"tooltip": "参考图 3，提示词中用 image3 引用"}),
                "image4": ("IMAGE", {"tooltip": "参考图 4"}),
                "image5": ("IMAGE", {"tooltip": "参考图 5"}),
                "image6": ("IMAGE", {"tooltip": "参考图 6"}),
                "image7": ("IMAGE", {"tooltip": "参考图 7"}),
                "reference_video": ("IMAGE", {"tooltip": "参考视频（ads2v/视频插入用）"}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "参考图/视频最大长边尺寸"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("正面条件", "负面条件", "latent")
    FUNCTION = "execute"
    CATEGORY = "AICoser-BerniniTools"

    def execute(
        self,
        clip, vae, width, height, length, batch_size,
        task_type, prompt, negative_prompt, use_default_neg,
        source_video=None,
        image0=None, image1=None, image2=None, image3=None,
        image4=None, image5=None, image6=None, image7=None,
        reference_video=None,
        ref_max_size=848,
    ):
        task_key = _parse_task(task_type)

        # --- 1. 文本编码 ---
        working_prompt = (prompt or "").strip()
        sys_prompt = _get_system_prompt(task_key)
        full_prompt = sys_prompt + " " + working_prompt if working_prompt else sys_prompt

        neg_text = (negative_prompt or "").strip()
        if not neg_text and use_default_neg:
            neg_text = DEFAULT_NEG_PROMPT

        from nodes import CLIPTextEncode
        _encoder = CLIPTextEncode()
        positive = _encoder.encode(clip, full_prompt)[0]
        negative = _encoder.encode(clip, neg_text)[0]

        # --- 2. 空 latent ---
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=mm.intermediate_device(),
        )

        # --- 3. 构造 context_latents ---
        context = []

        if source_video is not None:
            vid = common_upscale(
                source_video[:length, :, :, :3].movedim(-1, 1),
                width, height, "area", "center",
            ).movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))
            log.info("[BerniniTools] 已编码源视频: %d 帧 %dx%d", vid.shape[0], width, height)

        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)
            context.append(vae.encode(ref_vid[:, :, :, :3]))
            log.info("[BerniniTools] 已编码参考视频: %d 帧", ref_vid.shape[0])

        wired = [image0, image1, image2, image3, image4, image5, image6, image7]
        ref_images = [img for img in wired if img is not None]

        for idx, ref_img in enumerate(ref_images):
            for frame_idx in range(ref_img.shape[0]):
                img = _resize_long_edge(ref_img[frame_idx:frame_idx + 1], ref_max_size)
                context.append(vae.encode(img[:, :, :, :3]))
            log.info("[BerniniTools] 已编码参考图%d: %d 帧", idx, ref_img.shape[0])

        # --- 4. 附加 context_latents ---
        if context:
            if node_helpers is not None:
                positive = node_helpers.conditioning_set_values(
                    positive, {"context_latents": context}
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"context_latents": context}
                )
            else:
                for cond_list in [positive, negative]:
                    for item in cond_list:
                        item[1]["context_latents"] = context
            log.info("[BerniniTools] 任务 '%s': 已附加 %d 个 context 流", task_key, len(context))

        return (positive, negative, {"samples": latent})
