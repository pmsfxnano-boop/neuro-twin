"""Disease-specific literature evidence registry for candidate domain interactions.

Records are intentionally non-causal and non-parameterized. They cannot
instantiate a network or promote a coefficient into an estimated parameter.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro_twin.network.model import DOMAIN_NAMES


Disease = Literal["alzheimer", "parkinson"]
Domain = Literal["P", "I", "M", "N", "Q", "Pr"]
DirectionClass = Literal[
    "undirected_interaction",
    "directed_candidate",
    "reciprocal_candidate",
]
EvidenceContext = Literal[
    "review_synthesis",
    "human_genetics",
    "human_pathology",
    "human_biomarker",
    "human_longitudinal",
    "preclinical_model",
    "in_vitro",
]


class EvidenceCitation(BaseModel):
    """Bibliographic identity for one traceable evidence source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=8)
    publication_year: int = Field(ge=1900, le=2100)
    doi: str | None = None
    pmid: str | None = None
    url: str = Field(min_length=8)

    @model_validator(mode="after")
    def citation_has_identifier(self) -> "EvidenceCitation":
        if not self.doi and not self.pmid:
            raise ValueError("citation must include a DOI or PMID")
        if not self.url.startswith("https://"):
            raise ValueError("citation URL must use HTTPS")
        return self


class MechanisticInteractionRecord(BaseModel):
    """Curated literature-level interaction between two latent domains.

    Candidate directions are hypotheses, not causal edges. All records are
    ineligible for parameter estimation until an independent evidence review.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    interaction_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{3,80}$")
    disease: Disease
    domain_a: Domain
    domain_b: Domain
    direction_class: DirectionClass
    candidate_directions: tuple[tuple[Domain, Domain], ...] = ()
    mechanism_summary: str = Field(min_length=30)
    evidence_context: tuple[EvidenceContext, ...] = Field(min_length=1)
    references: tuple[EvidenceCitation, ...] = Field(min_length=1)
    limitations: str = Field(min_length=30)
    record_status: Literal["literature_candidate_only"] = "literature_candidate_only"
    sign_constraint: Literal["none"] = "none"
    model_eligible: Literal[False] = False

    @model_validator(mode="after")
    def directions_match_interaction(self) -> "MechanisticInteractionRecord":
        if self.domain_a == self.domain_b:
            raise ValueError("an interaction must connect two different domains")
        pair = {self.domain_a, self.domain_b}
        directions = set(self.candidate_directions)
        if any(
            len(direction) != 2 or direction[0] == direction[1]
            or set(direction) != pair
            for direction in directions
        ):
            raise ValueError("candidate directions must connect domain_a and domain_b")
        if len(directions) != len(self.candidate_directions):
            raise ValueError("candidate_directions must not contain duplicates")
        if self.direction_class == "undirected_interaction" and directions:
            raise ValueError("undirected interactions cannot declare directed candidates")
        if self.direction_class == "directed_candidate" and len(directions) != 1:
            raise ValueError("directed_candidate requires exactly one candidate direction")
        if self.direction_class == "reciprocal_candidate":
            expected = {
                (self.domain_a, self.domain_b),
                (self.domain_b, self.domain_a),
            }
            if directions != expected:
                raise ValueError("reciprocal_candidate requires both directions")
        return self


class MechanisticEvidenceRegistry(BaseModel):
    """Versioned, evidence-only registry; not an executable causal graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    registry_status: Literal["literature_candidates_only"] = "literature_candidates_only"
    records: tuple[MechanisticInteractionRecord, ...]

    @model_validator(mode="after")
    def unique_interaction_ids(self) -> "MechanisticEvidenceRegistry":
        ids = [record.interaction_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("interaction_id values must be unique")
        return self

    @property
    def registry_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def records_for_disease(
        self,
        disease: Disease,
    ) -> tuple[MechanisticInteractionRecord, ...]:
        return tuple(record for record in self.records if record.disease == disease)

    def directional_candidates_for_disease(
        self,
        disease: Disease,
    ) -> tuple[tuple[str, str], ...]:
        """List directed hypotheses only; undirected records stay unresolved."""
        candidates = {
            (source, target)
            for record in self.records_for_disease(disease)
            for source, target in record.candidate_directions
        }
        return tuple(sorted(candidates, key=lambda edge: (DOMAIN_NAMES.index(edge[0]), DOMAIN_NAMES.index(edge[1]))))

    def assert_not_parameterized(self) -> None:
        """Guard used by research tooling to reject automatic model promotion."""
        if any(record.model_eligible for record in self.records):
            raise ValueError("literature registry cannot promote model-eligible edges")
