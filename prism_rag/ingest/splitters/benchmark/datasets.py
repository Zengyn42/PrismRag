"""Hand-crafted benchmark cases for splitter evaluation.

Each case represents a realistic vault-style markdown section of varying
complexity. The reference_knots provide a gold-standard decomposition
that an ideal splitter would produce.
"""

from __future__ import annotations

from prism_rag.ingest.splitters.base import Knot
from prism_rag.ingest.splitters.benchmark.dataset import BenchmarkCase


def load_benchmark_dataset() -> list[BenchmarkCase]:
    """Return the built-in benchmark dataset."""
    return [
        _case_simple_fact(),
        _case_architecture_decisions(),
        _case_procedure_steps(),
        _case_mixed_concept_fact(),
    ]


# ---------------------------------------------------------------------------
# Case 1: Simple fact paragraph
# ---------------------------------------------------------------------------


def _case_simple_fact() -> BenchmarkCase:
    section_text = (
        "Redis is an in-memory data structure store. It supports strings, "
        "hashes, lists, sets, and sorted sets. Redis achieves durability "
        "through RDB snapshots and AOF logs. The default port is 6379."
    )
    return BenchmarkCase(
        section_text=section_text,
        doc_context="Redis Overview",
        reference_knots=[
            Knot(text="Redis is an in-memory data structure store.", ontology_type="fact"),
            Knot(
                text="Redis supports strings, hashes, lists, sets, and sorted sets.",
                ontology_type="fact",
            ),
            Knot(
                text="Redis achieves durability through RDB snapshots and AOF logs.",
                ontology_type="fact",
            ),
            Knot(text="The default Redis port is 6379.", ontology_type="fact"),
        ],
        source="hand-crafted: simple fact paragraph",
    )


# ---------------------------------------------------------------------------
# Case 2: Multi-decision architecture note
# ---------------------------------------------------------------------------


def _case_architecture_decisions() -> BenchmarkCase:
    section_text = (
        "We chose PostgreSQL over MySQL for the primary datastore because "
        "it has better JSON support and window functions. The read replicas "
        "will use streaming replication with a 30-second lag threshold.\n\n"
        "For the cache layer, we decided on Redis Cluster (3 shards) instead "
        "of Memcached because we need sorted sets for leaderboard queries. "
        "Cache TTL is set to 5 minutes for user profiles and 1 hour for "
        "static config."
    )
    return BenchmarkCase(
        section_text=section_text,
        doc_context="Backend Architecture Decisions — 2026-03",
        reference_knots=[
            Knot(
                text="PostgreSQL was chosen over MySQL for the primary datastore because of better JSON support and window functions.",
                ontology_type="decision",
            ),
            Knot(
                text="Read replicas use streaming replication with a 30-second lag threshold.",
                ontology_type="decision",
            ),
            Knot(
                text="Redis Cluster (3 shards) was chosen over Memcached for the cache layer because sorted sets are needed for leaderboard queries.",
                ontology_type="decision",
            ),
            Knot(
                text="Cache TTL is 5 minutes for user profiles and 1 hour for static config.",
                ontology_type="fact",
            ),
        ],
        source="hand-crafted: multi-decision architecture note",
    )


# ---------------------------------------------------------------------------
# Case 3: Procedure with steps
# ---------------------------------------------------------------------------


def _case_procedure_steps() -> BenchmarkCase:
    section_text = (
        "To deploy the staging environment:\n"
        "1. Pull the latest `main` branch and run `make build`.\n"
        "2. Run database migrations with `alembic upgrade head`.\n"
        "3. Set the `DEPLOY_ENV=staging` environment variable.\n"
        "4. Execute `./deploy.sh staging` and wait for the health check "
        "to return 200.\n"
        "5. Verify logs in Grafana dashboard under the staging namespace."
    )
    return BenchmarkCase(
        section_text=section_text,
        doc_context="Deployment Runbook",
        reference_knots=[
            Knot(
                text="To deploy the staging environment, first pull the latest main branch and run make build.",
                ontology_type="procedure",
            ),
            Knot(
                text="Run database migrations with alembic upgrade head.",
                ontology_type="procedure",
            ),
            Knot(
                text="Set the DEPLOY_ENV=staging environment variable.",
                ontology_type="procedure",
            ),
            Knot(
                text="Execute ./deploy.sh staging and wait for the health check to return 200.",
                ontology_type="procedure",
            ),
            Knot(
                text="Verify logs in Grafana dashboard under the staging namespace.",
                ontology_type="procedure",
            ),
        ],
        source="hand-crafted: procedure with steps",
    )


# ---------------------------------------------------------------------------
# Case 4: Mixed concept + fact section
# ---------------------------------------------------------------------------


def _case_mixed_concept_fact() -> BenchmarkCase:
    section_text = (
        "Event sourcing stores every state change as an immutable event in "
        "an append-only log. This enables full audit trails and temporal "
        "queries. Unlike CRUD, it never overwrites data.\n\n"
        "Our implementation uses Kafka as the event store with a 30-day "
        "retention policy. Each aggregate has its own topic. Snapshots are "
        "taken every 100 events to keep replay times under 500ms."
    )
    return BenchmarkCase(
        section_text=section_text,
        doc_context="Event Sourcing Architecture",
        reference_knots=[
            Knot(
                text="Event sourcing stores every state change as an immutable event in an append-only log.",
                ontology_type="concept",
            ),
            Knot(
                text="Event sourcing enables full audit trails and temporal queries.",
                ontology_type="concept",
            ),
            Knot(
                text="Unlike CRUD, event sourcing never overwrites data.",
                ontology_type="concept",
            ),
            Knot(
                text="The event sourcing implementation uses Kafka as the event store with a 30-day retention policy.",
                ontology_type="fact",
            ),
            Knot(
                text="Each aggregate has its own Kafka topic.",
                ontology_type="fact",
            ),
            Knot(
                text="Snapshots are taken every 100 events to keep replay times under 500ms.",
                ontology_type="fact",
            ),
        ],
        source="hand-crafted: mixed concept + fact section",
    )
