---
knowledge_id: KNOW-000032
title: PrismRag 多模态 Embedding — Text-first 与跨模态架构
ontology_type: concept
atomized_from: 设计细节/PrismRag Multimodal Embedding — 设计方向与路线图.md
status: active
---
PrismRag 设计了两层 embedding 方案：

**层次 1 (Text-first)**：将多模态内容转为纯文本后用文字 embedding 模型处理。图片通过 gemma4 生成视觉描述，音频通过 Whisper 转录，PDF 直接提取文字，统一用 gemini-embedding-2-preview 生成向量。

**层次 2 (真正跨模态)**：使用 nomic-embed-vision 模型将文字和图片映射到同一个 768 维向量空间，实现「以文搜图」和「以图搜文」。目前仅 PDF 文字提取已实装，图片和音频方案尚未实现。
