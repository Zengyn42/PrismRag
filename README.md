# PrismRag

> 无垠智穹 · 多模态 RAG 系统 · Zengyn42
> 从 NimbusVault（Obsidian 知识库）抽取知识，通过 MCP Server 对外提供跨模态语义检索。

**状态**：🚧 设计完成，待开发

---

## 简介

PrismRag 是无垠智穹集团的多模态 RAG（Retrieval-Augmented Generation）系统。

- **数据源**：[Zengyn42/NimbusVault](https://github.com/Zengyn42/NimbusVault) — Obsidian 知识库（Markdown + 附件）
- **Embedding**：Gemini Embedding 2（2026.3.10 发布，原生多模态）
- **存储**：LanceDB（嵌入式，统一 chunks / atoms / relations）
- **接口**：MCP Server（标准协议，集团所有 Agent 可调用）

名字取自 **Prism**（棱镜）—— 把一次查询分散到多模态检索空间，就像棱镜把光分散成光谱。

## 核心设计

| 维度 | 选型 | 理由 |
|---|---|---|
| 知识源 | Obsidian Vault | 双向链接 = 天然知识图谱 |
| Embedding | Gemini Embedding 2 | 原生多模态、统一向量空间、免费额度 + 低成本付费 |
| 存储 | LanceDB | 零部署、嵌入式、单存储统一 chunks / atoms / relations、内置 Hybrid Search |
| 部署 | MCP Server | 标准化工具协议、所有 Agent 可调用 |

### 设计原则

1. **务实精简** — Phase 1 只做必要功能，拒绝过度工程
2. **隐私优先** — 默认要求付费层 API，免费层需强制确认风险
3. **幂等可恢复** — 同步过程可中断可恢复，Checkpoint + 精确清理保证一致性
4. **多模态原生** — 充分利用 Gemini Embedding 2 的跨模态能力
5. **六空间罗盘** — 王延章六空间理论（K/S/D/F/I/O）作为设计语言，指导知识建模
6. **零部署** — `pip install` 即可启动，用户无需 Docker

### 完整架构文档

详细的架构设计、表结构、ADR、GraphRAG Phase 2 方案等，见 NimbusVault：

👉 [knowledge/Obsidian 多模态 RAG 系统架构设计.md](https://github.com/Zengyn42/NimbusVault/blob/master/knowledge/Obsidian多模态RAG系统架构设计.md)

本仓库内的 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 会持续同步简要版本。

## 目录结构

```
PrismRag/
├── prism_rag/              # 主包
│   ├── __init__.py
│   ├── config.py           # 配置（路径 / API key / 模型选择）
│   ├── ingest/             # NimbusVault → chunks → embeddings
│   ├── store/              # LanceDB 存储层
│   ├── retrieve/           # 检索层（Hybrid Search / GraphRAG）
│   ├── mcp_server/         # MCP Server 部署
│   └── cli.py              # CLI 入口（ingest / query / server）
├── tests/                  # 单元测试 + 集成测试
├── docs/                   # 本仓库的文档（指向 NimbusVault）
├── scripts/                # 一次性脚本（init_db、迁移等）
├── pyproject.toml          # 包元数据 + 依赖
├── .gitignore
└── README.md
```

**注**：目前 `prism_rag/` 只有 `__init__.py`，子模块在开发过程中按需创建。

## Quick Start

```bash
# 安装（开发模式）
git clone git@github.com:Zengyn42/PrismRag.git
cd PrismRag
pip install -e ".[dev]"

# 配置 Gemini API key
export GEMINI_API_KEY="..."

# 初次索引 NimbusVault（未实现）
# prism-rag ingest --vault ~/Foundation/Vault

# 启动 MCP Server（未实现）
# prism-rag serve
```

## 相关仓库

| Repo | 定位 |
|---|---|
| [Zengyn42/ZenithLoom](https://github.com/Zengyn42/ZenithLoom) | Agent 编排引擎（LangGraph 核心） |
| [Zengyn42/NimbusVault](https://github.com/Zengyn42/NimbusVault) | Obsidian 知识库（PrismRag 的数据源） |
| **Zengyn42/PrismRag** | **本仓库** |

## License

Proprietary — 内部使用。

---

*— Zengyn42 · 无垠智穹*
