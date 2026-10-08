"""Research-source registry.

URLs and access classes are recorded as metadata so the program can tell
public APIs from controlled research repositories. This file is not an
assertion that every source exposes every Neuro-Twin biomarker.
"""
from __future__ import annotations

SOURCE_REGISTRY = {
    "adni": {
        "display_name": "Alzheimer's Disease Neuroimaging Initiative",
        "class": "controlled_dataset",
        "transport": "LONI IDA",
        "entrypoint": "https://adni.loni.usc.edu/data-samples/adni-data/",
        "coverage": ["clinical", "cognition", "MRI", "PET", "biofluid", "genetics/omics"],
        "notes": "Application and ADNI Data Use Agreement required.",
    },
    "ad_knowledge_portal": {
        "display_name": "AD Knowledge Portal / Synapse",
        "class": "mixed_public_and_controlled",
        "transport": "Synapse Python client / CLI / portal",
        "entrypoint": "https://help.adknowledgeportal.org/apd/access-data",
        "coverage": ["clinical", "omics", "proteomics", "imaging", "multimodal"],
        "notes": "Open and controlled tiers; controlled human data require access request.",
    },
    "oasis": {
        "display_name": "OASIS-3 / OASIS-4",
        "class": "controlled_dataset",
        "transport": "NITRC-IR/XNAT + bulk download scripts",
        "entrypoint": "https://sites.wustl.edu/oasisbrains/home/oasis-3/",
        "coverage": ["MRI", "PET", "clinical", "cognition", "CSF/biomarker subsets"],
        "notes": "OASIS-3/OASIS-4 access requires NITRC account and approval/data-use terms.",
    },
    "openneuro": {
        "display_name": "OpenNeuro",
        "class": "public_api",
        "transport": "GraphQL API / CLI / DataLad / git",
        "entrypoint": "https://docs.openneuro.org/api.html",
        "coverage": ["BIDS neuroimaging datasets", "metadata", "snapshots"],
        "notes": "Public dataset search and metadata API; individual datasets may have access controls.",
    },
    "biostudies": {
        "display_name": "EMBL-EBI BioStudies",
        "class": "public_api",
        "transport": "REST API",
        "entrypoint": "https://www.ebi.ac.uk/biostudies/RH3R/help",
        "coverage": ["study metadata", "proteomics", "omics", "attachments"],
        "notes": "Public studies can be searched and retrieved programmatically.",
    },
    "ncbi_entrez_geo": {
        "display_name": "NCBI Entrez / GEO",
        "class": "public_api",
        "transport": "E-utilities",
        "entrypoint": "https://www.ncbi.nlm.nih.gov/books/NBK25497/",
        "coverage": ["GEO", "gene expression", "literature/provenance"],
        "notes": "Use API key/rate controls for sustained programmatic access.",
    },
}
