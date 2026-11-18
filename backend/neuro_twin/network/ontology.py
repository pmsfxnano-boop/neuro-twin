"""Hierarchical ontology contract for the six top-level Neuro-Twin domains.

Subconstructs are an evidence-organizing layer. They are not extra fitted
states and are not automatically averaged into a parent-domain score.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro_twin.network.evidence import EvidenceCitation
from neuro_twin.network.model import DOMAIN_NAMES, Domain


class DomainSubconstruct(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,64}$")
    label: str = Field(min_length=2)
    description: str = Field(min_length=20)


class DomainOntologyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: Domain
    construct_name: str = Field(min_length=2)
    operational_definition: str = Field(min_length=40)
    working_polarity: str = Field(min_length=10)
    subconstructs: tuple[DomainSubconstruct, ...] = Field(min_length=1)
    overlap_cautions: tuple[str, ...] = Field(min_length=1)
    not_equivalent_to: tuple[str, ...] = Field(min_length=1)
    aggregation_status: Literal["not_operationalized"] = "not_operationalized"
    citations: tuple[EvidenceCitation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_subconstruct_codes(self) -> "DomainOntologyEntry":
        codes = [item.code for item in self.subconstructs]
        if len(codes) != len(set(codes)):
            raise ValueError(f"subconstruct codes must be unique in domain {self.domain}")
        return self


class NeuroDomainOntology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    ontology_status: Literal["construct_inventory_not_measurement_model"] = (
        "construct_inventory_not_measurement_model"
    )
    entries: tuple[DomainOntologyEntry, ...]

    @model_validator(mode="after")
    def complete_unique_domain_set(self) -> "NeuroDomainOntology":
        domains = [entry.domain for entry in self.entries]
        if len(domains) != len(set(domains)):
            raise ValueError("each top-level domain must appear exactly once")
        if set(domains) != set(DOMAIN_NAMES):
            raise ValueError("ontology must define exactly P, I, M, N, Q and Pr")
        if tuple(domains) != DOMAIN_NAMES:
            raise ValueError("ontology entries must follow the canonical state-vector order")
        return self

    def get(self, domain: Domain) -> DomainOntologyEntry:
        for entry in self.entries:
            if entry.domain == domain:
                return entry
        raise KeyError(domain)

    @property
    def state_vector_order(self) -> tuple[str, ...]:
        return DOMAIN_NAMES
