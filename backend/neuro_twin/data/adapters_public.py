"""Concrete adapters for current public programmatic research services.

They are discovery/metadata-first. Raw files are not interpreted as clinical
measurements until a versioned source-specific parser creates canonical
Observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from neuro_twin.data.adapters import FetchRequest, SourceAdapter
from neuro_twin.data.http import HTTPClient
from neuro_twin.data.schema import AccessTier, Observation, ObservationKind, Provenance


class OpenNeuroAdapter(SourceAdapter):
    name = "openneuro"
    access_tier = AccessTier.PUBLIC_API
    GRAPHQL = "https://openneuro.org/crn/graphql"

    def __init__(self, graphql_url: str | None = None, client: HTTPClient | None = None):
        self.client = client or HTTPClient(graphql_url or self.GRAPHQL)

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.client.post("", json={"query": query, "variables": variables or {}}).json()
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        return payload.get("data", {})

    def discover(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        if request.dataset_id:
            query = """
            query Dataset($id: String!) {
              dataset(id: $id) { id name latestSnapshot { tag created hexsha } }
            }
            """
            dataset = self._graphql(query, {"id": request.dataset_id}).get("dataset")
            if dataset:
                yield dataset
            return

        q = (request.query or {})
        first = int(q.get("first", 20))
        after = q.get("after")
        query = """
        query Datasets($first: Int, $after: String, $public: Boolean) {
          datasets(first: $first, after: $after, filterBy: {public: $public}) {
            edges { cursor node { id name publishDate latestSnapshot { tag created hexsha } } }
            pageInfo { hasNextPage endCursor }
          }
        }
        """
        data = self._graphql(query, {"first": first, "after": after, "public": True})
        payload = data.get("datasets", {})
        for edge in payload.get("edges", []):
            node = edge.get("node") or {}
            node["_cursor"] = edge.get("cursor")
            yield node

    def fetch(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        yield from self.discover(request)

    def normalize(self, raw: dict[str, Any]) -> Observation:
        now = datetime.now(timezone.utc)
        dataset_id = str(raw["id"])
        return Observation(
            subject_id="__dataset__",
            event_id=dataset_id,
            acquisition_time=now,
            ingest_time=now,
            source=self.name,
            modality="dataset_metadata",
            observation_kind=ObservationKind.CLINICAL,
            feature="dataset_name",
            value=0.0,
            provenance=Provenance(
                source_name=self.name,
                source_record_id=dataset_id,
                retrieval_uri=self.GRAPHQL,
                access_tier=self.access_tier,
            ),
            extra={"dataset_id": dataset_id, "dataset_name": raw.get("name"), "latest_snapshot": raw.get("latestSnapshot")},
        )

    def snapshot_files(self, dataset_id: str, tag: str, recursive: bool = True) -> list[dict[str, Any]]:
        query = """
        query SnapshotFiles($datasetId: String!, $tag: String!, $recursive: Boolean!) {
          snapshot(datasetId: $datasetId, tag: $tag) {
            id tag files(recursive: $recursive) { id filename size directory annexed }
          }
        }
        """
        payload = self._graphql(query, {"datasetId": dataset_id, "tag": tag, "recursive": recursive})
        return payload.get("snapshot", {}).get("files", [])


class BioStudiesAdapter(SourceAdapter):
    name = "biostudies"
    access_tier = AccessTier.PUBLIC_API
    API = "https://www.ebi.ac.uk/biostudies/api/v1"

    def __init__(self, base_url: str | None = None, client: HTTPClient | None = None):
        self.client = client or HTTPClient(base_url or self.API)

    def discover(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        q = request.query or {}
        params = {"query": q.get("q", q.get("query", "*")), "pageSize": int(q.get("pageSize", 100))}
        if q.get("pagination"):
            params["pagination"] = q["pagination"]
        if q.get("cursor"):
            params["cursor"] = q["cursor"]
        response = self.client.get("search", params=params)
        payload = response.json()
        for result in payload.get("studies", payload.get("hits", [])):
            yield result

    def search_page(self, query: str, *, page_size: int = 1000, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "pagination": "cursor", "pageSize": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        return self.client.get("search", params=params).json()

    def iter_search(self, query: str, *, page_size: int = 1000):
        cursor: str | None = None
        while True:
            page = self.search_page(query, page_size=page_size, cursor=cursor)
            for item in page.get("studies", page.get("hits", [])):
                yield item
            cursor = page.get("nextCursor")
            if not cursor:
                break

    def fetch(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        yield from self.discover(request)

    def get_study(self, accession: str) -> dict[str, Any]:
        return self.client.get(f"studies/{accession}").json()

    def get_study_info(self, accession: str) -> dict[str, Any]:
        return self.client.get(f"studies/{accession}/info").json()

    def normalize(self, raw: dict[str, Any]) -> Observation:
        now = datetime.now(timezone.utc)
        accession = str(raw.get("accession", raw.get("accno", raw.get("id", "unknown"))))
        return Observation(
            subject_id="__study__",
            event_id=accession,
            acquisition_time=now,
            ingest_time=now,
            source=self.name,
            modality="study_metadata",
            observation_kind=ObservationKind.CLINICAL,
            feature="study_record",
            value=0.0,
            provenance=Provenance(
                source_name=self.name,
                source_record_id=accession,
                retrieval_uri=f"{self.API}/studies/{accession}",
                access_tier=self.access_tier,
            ),
            extra={"accession": accession, "title": raw.get("title", "")},
        )


class NCBIEntrezAdapter(SourceAdapter):
    name = "ncbi_entrez_geo"
    access_tier = AccessTier.PUBLIC_API
    API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str | None = None, tool: str = "neuro-twin", email: str | None = None, client: HTTPClient | None = None):
        self.api_key = api_key
        self.tool = tool
        self.email = email
        self.client = client or HTTPClient(self.API)

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": self.tool, **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def esearch(self, *, term: str, db: str = "gds", retmax: int = 20, retstart: int = 0) -> dict[str, Any]:
        response = self.client.get("esearch.fcgi", params=self._params({
            "db": db, "term": term, "retmode": "json", "retmax": retmax, "retstart": retstart,
        }))
        return response.json()["esearchresult"]

    def esummary(self, *, ids: list[str], db: str = "gds") -> dict[str, Any]:
        response = self.client.get("esummary.fcgi", params=self._params({
            "db": db, "id": ",".join(ids), "retmode": "json",
        }))
        return response.json()["result"]

    def efetch_text(self, *, ids: list[str], db: str = "gds", rettype: str = "full", retmode: str = "text") -> str:
        response = self.client.get("efetch.fcgi", params=self._params({
            "db": db, "id": ",".join(ids), "rettype": rettype, "retmode": retmode,
        }))
        return response.text

    def discover(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        q = request.query or {}
        result = self.esearch(
            term=q.get("term", "Alzheimer[Title/Abstract]"),
            db=q.get("db", "gds"),
            retmax=int(q.get("retmax", 20)),
            retstart=int(q.get("retstart", 0)),
        )
        db = q.get("db", "gds")
        term = q.get("term", "Alzheimer[Title/Abstract]")
        for uid in result.get("idlist", []):
            yield {"db": db, "uid": uid, "count": result.get("count"), "term": term}

    def fetch(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        yield from self.discover(request)

    def normalize(self, raw: dict[str, Any]) -> Observation:
        now = datetime.now(timezone.utc)
        uid = str(raw["uid"])
        return Observation(
            subject_id="__record__",
            event_id=uid,
            acquisition_time=now,
            ingest_time=now,
            source=self.name,
            modality=str(raw.get("db", "entrez")),
            observation_kind=ObservationKind.CLINICAL,
            feature="database_record",
            value=0.0,
            provenance=Provenance(
                source_name=self.name,
                source_record_id=uid,
                retrieval_uri=f"{self.API}/esearch.fcgi",
                access_tier=self.access_tier,
            ),
            extra=raw,
        )
