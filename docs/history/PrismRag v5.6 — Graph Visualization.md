---
title: PrismRag v5.6 — Graph Visualization
created: 2026-05-17
updated: 2026-05-19
tags: [prismrag, visualization, obsidian, design]
status: confirmed
---

# PrismRag v5.6 — Graph Visualization

## 一、背景

PrismRag 已有 `prism_rag/report/visualize.py`（pyvis + Barnes-Hut），能生成 HTML 知识图谱。现存问题：

1. **与 Obsidian 无集成**：HTML 是独立文件，节点点击无动作，不能跳转回笔记
2. **多 repo 合并太重**：code namespace 单独 3MB，合并后不可用

## 二、设计目标

| 目标 | 说明 |
|------|------|
| P1 每 repo 独立 HTML | 每个 namespace/graph 一个 graph.html，按需加载 |
| P2 Obsidian URI 集成 | vault 节点点击 → `obsidian://open?vault=...&file=...` |
| P3 Portal 节点 | 详细图里跨 namespace 引用显示为 portal，点击跳转目标 HTML |

> Federation meta-graph（多 namespace 汇总视图）推迟到 v6.0 Federation 阶段实现。

## 三、架构决策（辩论子图 2026-05-19 确认）

### 决策 1 — 技术栈：保留 pyvis + HTML 后处理注入

**结论：保留 pyvis，后处理注入自定义 JS。**

理由：
- pyvis 生成的 HTML 中 `network` 是全局变量（模板第 265 行），可在 `</body>` 前直接注入 `<script>` 引用
- `drawGraph()` 是同步调用（模板第 554 行），不在 `window.onload` 内，DOM 生命周期无问题
- `net.add_node()` 支持任意 kwargs，自定义属性（如 `obsidian_uri`）不会丢失
- `stabilizationIterationsDone` 事件已在模板中使用（第 542 行），可复用做 hash 定位的触发点
- 注入量极小（~30 行 JS），用 `html.replace("</body>", "<script>...</script></body>")` 实现

**否决方案**：切换 D3.js / 原生 vis.js — 重写成本高（数百行），与现有 pyvis API 不兼容，收益不值。

### 决策 2 — Obsidian 集成深度：浅层 URI 协议

**结论：浅层，生成静态 HTML + `obsidian://` URI 协议跳转。**

理由：
- `obsidian://open?vault=NimbusVault&file=<encoded_path>` 在 Obsidian 本地运行时直接生效
- 零额外依赖，不需要开发和维护 Obsidian 插件
- ROI 最高：一行 JS 实现节点点击跳转

**否决方案**：Obsidian 插件 — 开发周期长，需独立发布和更新，与 PrismRag 主仓库耦合重。

### 决策 3 — 触发时机：按需命令触发

**结论：按需触发，`prism visualize [--namespace X]`。**

理由：
- atomize_apply 后自动重生成会拖慢每次 ingest（图生成对大图有 CPU 开销）
- 用户在 MCP 工作流中通常不需要立即看图

**接口设计**：
```bash
prism visualize                    # 生成第一个 namespace 的 graph.html
prism visualize --namespace nimbus # 只生成 nimbus namespace
```

MCP tool `generate_graph(namespace="")` 供 Jei 调用。

### 决策 4 — Portal 节点 URL hash 导航

**结论：`graph.html#<node-id>` 格式，在 `stabilizationIterationsDone` 事件中读取 hash 并 focus 节点。**

实现模式：
```javascript
// 注入到生成的 graph.html 中
network.on("stabilizationIterationsDone", function() {
    var hash = window.location.hash.slice(1);
    if (hash && nodes.get(hash)) {
        network.focus(hash, { scale: 1.5, animation: true });
        network.selectNodes([hash]);
    }
});
```

Portal 节点点击：
```javascript
network.on("click", function(params) {
    if (params.nodes.length > 0) {
        var data = nodes.get(params.nodes[0]);
        if (data.portal_href) {
            window.open(data.portal_href, "_blank");
        } else if (data.obsidian_uri) {
            window.location.href = data.obsidian_uri;
        }
    }
});
```

## 四、详细实现方案

### P1 — 每 repo 独立 HTML

`FederatedGraph` 已按 namespace 分开存储，对每个 `source.graph` 分别调用 `generate_html()`：

```python
for source in federated.sources:
    output = source.data_dir / "graph.html"
    generate_html(source.graph, output, vault_name=source.namespace)
```

生成路径：
```
data/nimbus/graph.html   ← NimbusVault
data/code/graph.html     ← 代码库
```

### P2 — Obsidian URI 注入

`generate_html()` 后处理：在写入文件前替换 `</body>`：

```python
OBSIDIAN_JS = """
<script>
network.on("click", function(params) {
    if (params.nodes.length === 0) return;
    var data = nodes.get(params.nodes[0]);
    if (data.portal_href) {
        window.open(data.portal_href, "_blank");
    } else if (data.obsidian_uri) {
        window.location.href = data.obsidian_uri;
    }
});
network.on("stabilizationIterationsDone", function() {
    var hash = window.location.hash.slice(1);
    if (hash && nodes.get(hash)) {
        network.focus(hash, { scale: 1.5, animation: true });
        network.selectNodes([hash]);
    }
});
</script>
</body>"""

html_content = html_content.replace("</body>", OBSIDIAN_JS)
```

节点添加时携带 `obsidian_uri`：

```python
# vault 类节点（kind=note/knowledge）
vault_name = "NimbusVault"  # 从 config 读取
file_path = urllib.parse.quote(node.get("file_path", ""), safe="")
obsidian_uri = f"obsidian://open?vault={vault_name}&file={file_path}"
net.add_node(node_id, ..., obsidian_uri=obsidian_uri)
```

### P3 — Portal 节点渲染

跨 namespace 引用节点（`kind='context_ref'` 或带 `mentions_symbol` 的代码引用）渲染为六边形：

```python
if node.get("kind") == "context_ref" or node.get("cross_namespace"):
    net.add_node(node_id,
                 label=f"⬡ {short_label}",
                 shape="hexagon",
                 color="#F5A623",
                 portal_href=f"../{target_ns}/graph.html#{target_id}",
                 title=f"Portal → {target_ns}:{target_id}")
```

## 五、实现范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `prism_rag/report/visualize.py` | 修改 | 后处理 JS 注入；obsidian_uri + portal_href 节点属性；portal 节点六边形渲染 |
| `prism_rag/cli/commands/visualize.py` | 新建 | `prism visualize` CLI 命令 |
| `prism_rag/cli/main.py` | 修改 | 注册 visualize 子命令 |
| `prism_rag/mcp_server/server.py` | 修改 | 新增 `generate_graph(namespace)` MCP tool |
| `tests/test_visualize.py` | 新建/修改 | 验证 JS 注入、obsidian_uri 生成、portal 节点属性 |

**工作量估算**：约半天。

## 六、依赖

- 依赖 v5.5 中 `CONTEXT_REF` 节点定义（portal 节点渲染识别跨 namespace 引用）
- Obsidian 本地运行时 URI 协议生效；浏览器直接打开 HTML 时 obsidian:// 无效（设计范围内，可接受）
- Federation meta-graph（多 namespace 汇总视图）→ 推迟到 v6.0
