---
tags:
- '#PrismRag'
- '#GraphRAG'
- '#Embedding'
- '#Multimodal'
- '#Ollama'
- '#architecture'
status: deferred
created: 2026-04-27
atomized_nodes:
- KNOW-000009
- KNOW-000015
- KNOW-000030
- KNOW-000008
- KNOW-000011
- KNOW-000013
- KNOW-000014
- KNOW-000029
- KNOW-000010
- KNOW-000031
---

# PrismRag 多模态 Embedding — 设计方向与路线图

> 状态：**设计已明确，实现暂缓**。Pass 2 的图片/音频提取位置已预留，待需要时按本文路线实施。

## 背景

当前 [[PrismRag-v4.0-设计文档]] 的 Pass 3 只支持文字 embedding（`gemini-embedding-2-preview`）。  
Pass 2 的 `extract_image()` 和 `extract_audio()` 均为 `raise NotImplementedError`，是有意留的扩展口。

触发本设计的问题：**如果 vault 里有图片/PDF/音频，embedding 应该怎么做？**

---

## 两个层次的"多模态"

### 层次 1 — Text-first（够用，推荐优先实施）

```
图片 ──→ 视觉模型描述 ──→ 文字 ──→ 文字 embedding
音频 ──→ Whisper 转录  ──→ 文字 ──→ 文字 embedding
PDF  ──→ pypdf 提取    ──→ 文字 ──→ 文字 embedding  ✅ 已实现
```

图片描述用本地 **gemma4**（已装），音频转录用 **Whisper**（按需安装）。  
结果存入 `node.content`，Pass 3 正常 embed 文字，**不需要新 pull 任何模型**。

### 层次 2 — 真正的跨模态（图片 ↔ 文字在同一向量空间）

```
文字 ─┐
      ├─→ 同一 768-dim 向量空间 → 可以"用文字搜图片"、"用图片搜文字"
图片 ─┘
```

需要专用跨模态 embedding 模型。

---

## Embedding 模型选型

| 模型 | 运行方式 | 维度 | 中文质量 | 图片支持 |
|---|---|---|---|---|
| `gemini-embedding-2-preview` | Gemini API | 768（Matryoshka 可调） | ⭐⭐⭐⭐⭐ | ❌ 仅文字 |
| `bge-m3` | `ollama pull bge-m3` | 1024 | ⭐⭐⭐⭐ | ❌ 仅文字 |
| `nomic-embed-text` | `ollama pull nomic-embed-text` | 768 | ⭐⭐⭐ | ❌ 仅文字 |
| `nomic-embed-vision` | `ollama pull nomic-embed-vision` | 768 | ⭐⭐⭐ | ✅ 文字+图片同空间 |
| CLIP / OpenCLIP | HuggingFace 本地 | 512/768 | ⭐⭐ | ✅ 跨模态 |

**关键发现：Gemini Embedding 2 不支持图片 embedding。**  
反而 Ollama 的 `nomic-embed-vision` 能把文字和图片 embed 到同一个向量空间——这是 Ollama 在跨模态场景比 Gemini 强的地方。

---

## 实施路线

### 路线 A — Text-first（短期，推荐先做）

```python
# prism_rag/ingest/media_extractor.py

def extract_image(path: Path) -> str:
    """用 gemma4 (Ollama) 生成图片描述，返回文字。"""
    import httpx
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = httpx.post("http://localhost:11434/api/generate", json={
        "model": "gemma4:e4b",
        "prompt": "请用中文详细描述这张图片的内容，包括所有文字、图表、结构信息。",
        "images": [img_b64],
        "stream": False,
    }, timeout=60)
    return resp.json().get("response", "")

def extract_audio(path: Path) -> str:
    """用 faster-whisper 转录音频，返回文字。"""
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cpu")
    segments, _ = model.transcribe(str(path), language="zh")
    return " ".join(s.text for s in segments)
```

Pass 3 不需要改动，bge-m3 或 Gemini 正常 embed 转录/描述文字。

### 路线 B — 真正跨模态（长期，按需）

```bash
ollama pull nomic-embed-vision  # 768-dim，文字+图片同空间
```

```python
# Pass 3 中对图片节点改用 nomic-embed-vision
def embed_image_node(path: Path) -> list[float]:
    import httpx, base64
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = httpx.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-vision",
        "input": img_b64,
    })
    return resp.json()["embeddings"][0]
```

文字节点继续用 bge-m3（768-dim），两者维度相同但**向量空间未对齐**——  
真正的跨模态搜索需要同一模型处理文字和图片（即统一用 nomic-embed-vision）。

---

## 决策记录

| 时间 | 决策 |
|---|---|
| 2026-04-27 | 当前 vault 以中文文字为主，多模态暂缓，优先补全文字 embedding（bge-m3 或 Gemini） |
| 2026-04-27 | 有图片/音频需求时，先走路线 A（Text-first + gemma4），无需新 pull 模型 |
| 2026-04-27 | 需要"以图搜图"或"以文搜图"时，再走路线 B（nomic-embed-vision） |

## 关联节点

- [[PrismRag-v4.0-设计文档]]
- [[PrismRag Phase 1 MVP 实现详情]]
- [[PrismRag Phase 2 实现详情]]
- [[Obsidian多模态RAG系统架构设计]]
