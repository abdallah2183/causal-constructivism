from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .graph import CausalGraph
from .models import Edge, EdgeType, GaussianBelief, Node, NodeType


SCHEMA_VERSION = 1


class SQLiteGraphStore:
    """Transactional, append-only snapshots for causal graphs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save(self, graph: CausalGraph, *, label: str | None = None) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO snapshots(label) VALUES (?)",
                (label,),
            )
            snapshot_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO nodes(
                    snapshot_id, node_id, name, node_type, prior_mean,
                    prior_variance, belief_mean, belief_variance,
                    evidence_mean, evidence_variance, modality, is_axiom,
                    metadata_json, version, created_at, updated_at,
                    deprecated_at, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._node_row(snapshot_id, node) for node in graph.nodes],
            )
            connection.executemany(
                """
                INSERT INTO edges(
                    snapshot_id, edge_id, source_id, target_id, edge_type,
                    weight, bias, noise_variance, confidence, learned,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._edge_row(snapshot_id, edge) for edge in graph.edges],
            )
            history_rows = []
            for owner_id, snapshots in graph.history_items():
                history_rows.extend(
                    self._history_row(snapshot_id, owner_id, index, node)
                    for index, node in enumerate(snapshots)
                )
            connection.executemany(
                """
                INSERT INTO node_history(
                    snapshot_id, owner_node_id, sequence, node_json
                ) VALUES (?, ?, ?, ?)
                """,
                history_rows,
            )
            return snapshot_id

    def load(self, snapshot_id: int | None = None) -> CausalGraph:
        connection = sqlite3.connect(self.path)
        try:
            connection.row_factory = sqlite3.Row
            if snapshot_id is None:
                row = connection.execute(
                    "SELECT MAX(id) AS id FROM snapshots"
                ).fetchone()
                snapshot_id = row["id"]
            if snapshot_id is None:
                raise KeyError("No graph snapshots exist")
            exists = connection.execute(
                "SELECT 1 FROM snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown graph snapshot: {snapshot_id}")

            graph = CausalGraph()
            for row in connection.execute(
                "SELECT * FROM nodes WHERE snapshot_id = ? ORDER BY rowid",
                (snapshot_id,),
            ):
                graph.restore_node(self._node_from_row(row))
            for row in connection.execute(
                "SELECT * FROM edges WHERE snapshot_id = ? ORDER BY rowid",
                (snapshot_id,),
            ):
                graph.restore_edge(self._edge_from_row(row))

            grouped: dict[str, list[Node]] = {}
            for row in connection.execute(
                """
                SELECT owner_node_id, node_json
                FROM node_history
                WHERE snapshot_id = ?
                ORDER BY owner_node_id, sequence
                """,
                (snapshot_id,),
            ):
                grouped.setdefault(row["owner_node_id"], []).append(
                    self._node_from_json(row["node_json"])
                )
            for owner_id, snapshots in grouped.items():
                graph.restore_history(owner_id, snapshots)
            return graph
        finally:
            connection.close()

    def list_snapshots(self) -> tuple[tuple[int, str | None, str], ...]:
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute(
                "SELECT id, label, created_at FROM snapshots ORDER BY id"
            ).fetchall()
        finally:
            connection.close()
        return tuple((int(row[0]), row[1], row[2]) for row in rows)

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS schema_info(
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS nodes(
                    snapshot_id INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    prior_mean REAL NOT NULL,
                    prior_variance REAL NOT NULL,
                    belief_mean REAL NOT NULL,
                    belief_variance REAL NOT NULL,
                    evidence_mean REAL,
                    evidence_variance REAL,
                    modality TEXT NOT NULL,
                    is_axiom INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deprecated_at TEXT,
                    superseded_by TEXT,
                    PRIMARY KEY(snapshot_id, node_id),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS edges(
                    snapshot_id INTEGER NOT NULL,
                    edge_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    bias REAL NOT NULL,
                    noise_variance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    learned INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(snapshot_id, edge_id),
                    FOREIGN KEY(snapshot_id, source_id)
                        REFERENCES nodes(snapshot_id, node_id),
                    FOREIGN KEY(snapshot_id, target_id)
                        REFERENCES nodes(snapshot_id, node_id)
                );
                CREATE TABLE IF NOT EXISTS node_history(
                    snapshot_id INTEGER NOT NULL,
                    owner_node_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    node_json TEXT NOT NULL,
                    PRIMARY KEY(snapshot_id, owner_node_id, sequence),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_name
                    ON nodes(snapshot_id, name);
                CREATE INDEX IF NOT EXISTS idx_edges_source
                    ON edges(snapshot_id, source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target
                    ON edges(snapshot_id, target_id);
                """
            )
            row = connection.execute(
                "SELECT version FROM schema_info LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_info(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported graph database schema: {row[0]}"
                )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _node_row(snapshot_id: int, node: Node) -> tuple[Any, ...]:
        return (
            snapshot_id,
            node.id,
            node.name,
            node.node_type.value,
            node.prior.mean,
            node.prior.variance,
            node.belief.mean,
            node.belief.variance,
            node.evidence.mean if node.evidence else None,
            node.evidence.variance if node.evidence else None,
            node.modality,
            int(node.is_axiom),
            json.dumps(node.metadata, sort_keys=True),
            node.version,
            node.created_at.isoformat(),
            node.updated_at.isoformat(),
            node.deprecated_at.isoformat() if node.deprecated_at else None,
            node.superseded_by,
        )

    @staticmethod
    def _edge_row(snapshot_id: int, edge: Edge) -> tuple[Any, ...]:
        return (
            snapshot_id,
            edge.id,
            edge.source_id,
            edge.target_id,
            edge.edge_type.value,
            edge.weight,
            edge.bias,
            edge.noise_variance,
            edge.confidence,
            int(edge.learned),
            json.dumps(edge.metadata, sort_keys=True),
            edge.created_at.isoformat(),
        )

    @classmethod
    def _history_row(
        cls,
        snapshot_id: int,
        owner_id: str,
        sequence: int,
        node: Node,
    ) -> tuple[int, str, int, str]:
        payload = cls._node_row(0, node)[1:]
        return (
            snapshot_id,
            owner_id,
            sequence,
            json.dumps(payload),
        )

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> Node:
        evidence = (
            GaussianBelief(row["evidence_mean"], row["evidence_variance"])
            if row["evidence_mean"] is not None
            else None
        )
        return Node(
            id=row["node_id"],
            name=row["name"],
            node_type=NodeType(row["node_type"]),
            prior=GaussianBelief(row["prior_mean"], row["prior_variance"]),
            belief=GaussianBelief(row["belief_mean"], row["belief_variance"]),
            evidence=evidence,
            modality=row["modality"],
            is_axiom=bool(row["is_axiom"]),
            metadata=json.loads(row["metadata_json"]),
            version=int(row["version"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            deprecated_at=(
                datetime.fromisoformat(row["deprecated_at"])
                if row["deprecated_at"]
                else None
            ),
            superseded_by=row["superseded_by"],
        )

    @staticmethod
    def _edge_from_row(row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=row["weight"],
            bias=row["bias"],
            noise_variance=row["noise_variance"],
            confidence=row["confidence"],
            learned=bool(row["learned"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _node_from_json(payload: str) -> Node:
        values = json.loads(payload)
        (
            node_id,
            name,
            node_type,
            prior_mean,
            prior_variance,
            belief_mean,
            belief_variance,
            evidence_mean,
            evidence_variance,
            modality,
            is_axiom,
            metadata_json,
            version,
            created_at,
            updated_at,
            deprecated_at,
            superseded_by,
        ) = values
        return Node(
            id=node_id,
            name=name,
            node_type=NodeType(node_type),
            prior=GaussianBelief(prior_mean, prior_variance),
            belief=GaussianBelief(belief_mean, belief_variance),
            evidence=(
                GaussianBelief(evidence_mean, evidence_variance)
                if evidence_mean is not None
                else None
            ),
            modality=modality,
            is_axiom=bool(is_axiom),
            metadata=json.loads(metadata_json),
            version=int(version),
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            deprecated_at=(
                datetime.fromisoformat(deprecated_at)
                if deprecated_at
                else None
            ),
            superseded_by=superseded_by,
        )
