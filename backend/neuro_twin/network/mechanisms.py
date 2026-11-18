"""Typed Alzheimer mechanism-map registry.

The catalogue preserves mechanistic hypotheses and evidence provenance.
It is deliberately not an executable causal DAG and contains no fitted
coefficients, patient-level effect estimates, or automatic state mappings.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvidenceGrade = Literal["established", "strong", "moderate", "emerging", "uncertain"]
EvidenceType = Literal[
    "human_genetics",
    "longitudinal_human",
    "human_neuropathology",
    "clinical_intervention",
    "human_imaging",
    "human_biomarker",
    "in_vitro",
    "animal_model",
    "review_synthesis",
]
CausalStatus = Literal[
    "genetically_supported",
    "intervention_supported",
    "convergent_candidate",
    "association_only",
    "hypothesis_only",
]


class MechanismNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,80}$")
    label: str = Field(min_length=2)
    biological_scale: Literal[
        "genetic", "molecular", "organelle", "cellular", "circuit",
        "tissue", "systemic", "clinical"
    ]
    module: str = Field(min_length=2)
    biological_role: Literal[
        "susceptibility", "initiating_process", "clearance_process",
        "amplifier", "damage_substrate", "compensatory_process",
        "copathology", "phenotype"
    ]
    evidence_grade: EvidenceGrade
    description: str = Field(min_length=40)
    limitations: tuple[str, ...] = Field(min_length=1)
    source_refs: tuple[str, ...] = Field(min_length=1)


class MechanismEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,80}$")
    source_node: str = Field(min_length=3)
    target_node: str = Field(min_length=3)
    relationship: str = Field(min_length=5)
    directionality: Literal["directed_candidate", "bidirectional", "unresolved"]
    causal_status: CausalStatus
    evidence_grade: EvidenceGrade
    evidence_types: tuple[EvidenceType, ...] = Field(min_length=1)
    context: str = Field(min_length=20)
    limitations: tuple[str, ...] = Field(min_length=1)
    source_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def causal_claim_requires_matching_evidence(self) -> "MechanismEdge":
        if self.causal_status == "genetically_supported":
            if "human_genetics" not in self.evidence_types:
                raise ValueError("genetically_supported edges require human_genetics evidence")
        if self.causal_status == "intervention_supported":
            if "clinical_intervention" not in self.evidence_types:
                raise ValueError("intervention_supported edges require clinical_intervention evidence")
        if self.source_node == self.target_node:
            raise ValueError("self-edges are not allowed in the mechanism catalogue")
        return self


class AlzheimerMechanismCatalog(BaseModel):
    """Evidence-graded mechanism inventory, not a validated causal network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    disease: Literal["alzheimer"] = "alzheimer"
    catalog_status: Literal[
        "research_hypothesis_registry_not_validated_causal_dag"
    ] = "research_hypothesis_registry_not_validated_causal_dag"
    scope_statement: str = Field(min_length=40)
    nodes: tuple[MechanismNode, ...] = Field(min_length=1)
    edges: tuple[MechanismEdge, ...]

    @model_validator(mode="after")
    def unique_nodes_and_valid_edges(self) -> "AlzheimerMechanismCatalog":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("mechanism node identifiers must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("mechanism edge identifiers must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source_node not in known or edge.target_node not in known:
                raise ValueError(
                    f"edge {edge.edge_id} references an unknown mechanism node"
                )
        return self

    def node(self, node_id: str) -> MechanismNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)
