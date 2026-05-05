from prism_rag.ingest.symbol_linker import _clear_existing_mentions
from prism_rag.store.graph import Edge, KnowledgeGraph, LifecycleClass, Node


def test_clear_keeps_anchored_mentions():
    g = KnowledgeGraph()
    g.add_node(Node(id="doc", label="doc"))
    g.add_node(Node(id="code::Foo", label="Foo"))
    g.add_node(Node(id="code::Bar", label="Bar"))
    g.add_edge(Edge(source="doc", target="code::Foo", relation="mentions_symbol",
                    lifecycle_class=LifecycleClass.DETERMINISTIC))
    g.add_edge(Edge(source="doc", target="code::Bar", relation="mentions_symbol",
                    lifecycle_class=LifecycleClass.ANCHORED))
    _clear_existing_mentions(g.g)
    assert not g.g.has_edge("doc", "code::Foo")
    assert g.g.has_edge("doc", "code::Bar")


def test_clear_skips_non_mentions_relations():
    g = KnowledgeGraph()
    g.add_node(Node(id="x", label="x"))
    g.add_node(Node(id="y", label="y"))
    g.add_edge(Edge(source="x", target="y", relation="links_to"))
    _clear_existing_mentions(g.g)
    assert g.g.has_edge("x", "y")
