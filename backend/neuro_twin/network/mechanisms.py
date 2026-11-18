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


class OriginArchitectureHypothesis(BaseModel):
    """One falsifiable candidate architecture for Alzheimer disease origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,80}$")
    label: str = Field(min_length=3)
    primary_hypothesis: str = Field(min_length=60)
    candidate_early_nodes: tuple[str, ...] = Field(min_length=1)
    candidate_edge_ids: tuple[str, ...] = Field(min_length=1)
    discriminating_prediction: str = Field(min_length=60)
    falsification_criteria: tuple[str, ...] = Field(min_length=1)
    evidence_grade: EvidenceGrade
    limitations: tuple[str, ...] = Field(min_length=1)


class AlzheimerOriginArchitectureRegistry(BaseModel):
    """Competing origin hypotheses linked to the shared mechanism inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    disease: Literal["alzheimer"] = "alzheimer"
    registry_status: Literal[
        "competing_falsifiable_hypotheses_not_ranked"
    ] = "competing_falsifiable_hypotheses_not_ranked"
    architectures: tuple[OriginArchitectureHypothesis, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_architectures(self) -> "AlzheimerOriginArchitectureRegistry":
        identifiers = [item.architecture_id for item in self.architectures]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("architecture identifiers must be unique")
        return self

    def validate_against(
        self, catalog: AlzheimerMechanismCatalog
    ) -> "AlzheimerOriginArchitectureRegistry":
        if catalog.disease != self.disease:
            raise ValueError("origin architectures and mechanism catalog disease mismatch")
        node_ids = {item.node_id for item in catalog.nodes}
        edge_by_id = {item.edge_id: item for item in catalog.edges}
        for architecture in self.architectures:
            missing_nodes = set(architecture.candidate_early_nodes) - node_ids
            missing_edges = set(architecture.candidate_edge_ids) - set(edge_by_id)
            if missing_nodes:
                raise ValueError(
                    f"{architecture.architecture_id} references unknown nodes: {sorted(missing_nodes)}"
                )
            if missing_edges:
                raise ValueError(
                    f"{architecture.architecture_id} references unknown edges: {sorted(missing_edges)}"
                )
            for edge_id in architecture.candidate_edge_ids:
                edge = edge_by_id[edge_id]
                if edge.source_node not in node_ids or edge.target_node not in node_ids:
                    raise ValueError(f"{architecture.architecture_id} has invalid edge endpoints")
        return self
