# Neuro-Twin — metabolic-first evidence audit v1

**Status:** evidence audit from public human datasets and published secondary analyses.  
**Branch:** \`feat/six-domain-dynamic-network-v1\`  
**Scientific status:** hypothesis assessment only. No raw expression matrix has yet been ingested or independently reanalysed by Neuro-Twin in this audit.

## 1. Question under test

The working hypothesis is not merely that mitochondria are impaired in neurodegeneration. It is the stronger, falsifiable proposition:

> An early disturbance in cellular energy handling (glucose transport/glycolysis, pyruvate oxidation through PDH, TCA flux, respiratory-chain/OXPHOS capacity, redox balance and mitochondrial quality control) reduces neuronal resilience. Disease-specific cell, circuit, genetic and proteostasis contexts then route this shared vulnerability toward Alzheimer-type or Parkinson-type pathology.

This hypothesis must compete with, and may coexist with:

1. Amyloid production / clearance dominant (Alzheimer).
2. Tau / neuronal-vulnerability dominant (Alzheimer).
3. Alpha-synuclein / LRRK2 / endolysosomal-proteostasis dominant (Parkinson).
4. Neurovascular / perfusion / clearance dominant.
5. Glial / immune-state dominant.
6. Metabolic / mitochondrial dominant.
7. Mixed multi-hit / nonlinear threshold transition.

The model must distinguish **initiator**, **mediator**, **amplifier**, **compensatory response** and **consequence of cell loss**. A mitochondrial transcript signature alone cannot identify which role it plays.

## 2. Public human datasets located

All datasets below are discoverable from GEO without an ADNI/PPMI account. This means the data are publicly deposited; it does not mean all raw files are small or already parsed into Neuro-Twin's canonical observation schema.

| Disease / use | Dataset | Human material and size | Why it matters | Main caveat |
|---|---|---|---|---|
| Parkinson, cell-resolved primary set | [GSE184950](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184950) | Postmortem substantia nigra; 34 donors (10 controls, 6 PD, 18 PDD); raw filtered 10x matrices | Cell-specific states across control, PD and PDD; directly relevant to nigral vulnerability | Cross-sectional postmortem tissue; does not establish temporal order. GEO lists a 1.7 GB raw TAR |
| Parkinson, independent midbrain replication / best first pilot | [GSE157783](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157783) | Postmortem midbrain; 11 donors (6 controls, 5 idiopathic PD); about 41.4k nuclei after published QC | Independent midbrain atlas; GEO provides UMI matrix TAR (~123.3 MB) plus cell/gene metadata | Small donor cohort; postmortem; counts/nuclei vary with preprocessing and group labels must be checked against GEO sample records |
| Alzheimer, cell-resolved primary set | [GSE157827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157827) | Human prefrontal cortex; 21 donors (12 AD, 9 controls); 167,290 nuclei in the 2026 reanalysis | Donor-level pseudobulk and cell-type-specific expression | Small donor count; postmortem; some metabolic genes/cell-type combinations are not estimable |
| Alzheimer, independent early-vulnerable-region atlas | [GSE138852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852) | Entorhinal cortex; 12 donors (6 AD, 6 controls); 13,214 nuclei in the source study | Independent cell-resolved comparison in an early-vulnerable region | Only a small fraction of recovered nuclei are neurons in the later reanalysis; many neuronal gene/cell-type tests are not estimable |
| Alzheimer, late-stage multiomic atlas | [GSE174367](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174367) | Postmortem prefrontal cortex; original analyses report 12 AD and 8 controls; ~58.9k nuclei in one published reanalysis | Processed snRNA matrix (.h5, ~261 MB) and cell metadata allow a cell-type cross-check; includes paired chromatin context | Donor inclusion and nuclei count vary across reanalyses; reconcile GEO sample IDs and metadata before using as a formal replication |
| Alzheimer, large bulk discovery / coexpression | [GSE33000](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000) | Prefrontal cortex; 310 AD and 157 non-demented controls in the cited 2026 reanalysis | Large human donor cohort for pathway replication and coexpression | Bulk tissue; neuronal composition and disease severity can influence signals |
| Alzheimer, laser-captured neurons | [GSE5281](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5281) | 161 arrays, 87 AD and 74 controls, across six brain regions | Enriches neuronal material and supports region-specific comparisons | GEO donor identifiers are not recorded adequately for cross-region donor-level linkage; analyse within region |
| Alzheimer, temporal cortex replication | [GSE132903](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132903) | 195 arrays, 97 AD and 98 controls; one array per donor | Independent cohort for cortical energy-module replication | Bulk / platform-specific; gene coverage differs |
| Alzheimer, multi-region donor cohort | [GSE48350](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48350) | 253 arrays from 84 donors; 28 AD and 56 controls | Tests region-aware replication | Multiple regions per donor; must collapse to donor-level after within-region standardisation |
| Alzheimer / vascular dementia comparison | [GSE122063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063) | 136 arrays from 31 donors: 12 AD, 8 vascular dementia, 11 controls | Important disease-specificity comparison | Two regions and technical structure per donor; use donor-aware model |
| Parkinson, older bulk validation candidate | [GSE7621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7621) | Postmortem substantia nigra; 16 PD and 9 controls | Independent, smaller bulk expression comparison | 2007 Affymetrix series; tissue composition and legacy preprocessing; secondary validation only |
| Parkinson, regional bulk candidate | [GSE8397](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8397) | Medial/lateral substantia nigra and frontal cortex; 47 tissue samples, 94 A/B arrays | Region contrast and historical replication | Medial/lateral samples can come from the same cases; A/B chips are not independent donors |

### Longitudinal cohorts needed for temporal order

- **ADNI** provides longitudinal clinical, MRI, PET, fluid biomarker and genetic data, but individual-level data require application and acceptance of its Data Use Agreement through LONI IDA: [official access page](https://adni.loni.usc.edu/data-samples/adni-data/).
- **PPMI** provides longitudinal clinical, imaging, omics, sensor and biomarker data, but individual-level access requires an application and Data Use Agreement: [official access page](https://www.ppmi-info.org/access-data-specimens/download-data).
- **AMP-AD / AD Knowledge Portal** has open and controlled tiers; individual-level human data require additional access controls: [official access policy](https://help.adknowledgeportal.org/apd/access-data).

The DUA-governed cohorts are the right route for lead/lag modelling, but they are not treated as anonymously downloadable public tables. Any controlled data must be handled under the applicable agreement and an approved secure workflow; do not upload restricted participant-level data into an unapproved public-facing AI service.

## 3. First evidence extracted from published human-data analyses

This section records results reported by the cited studies. These are **published secondary analyses**, not Neuro-Twin's independent reprocessing of raw matrices.

### 3.1 Alzheimer: PDH–lipoylation–TCA signal

Zhang et al. (2026), DOI [10.1007/s12031-026-02598-7](https://doi.org/10.1007/s12031-026-02598-7), analysed public human cohorts including GSE157827, GSE33000, GSE5281, GSE132903, GSE122063 and GSE48350.

Reported findings:

- The single-nucleus discovery set comprised 167,290 nuclei from 21 prefrontal-cortex donors (12 AD, 9 controls).
- An independent entorhinal-cortex atlas, GSE138852 (6 AD, 6 controls; 13,214 nuclei), did not replicate a corrected gene–cell-type signal. PDHA1 was testable in only one cell type (oligodendrocytes) and was not significant; the low neuronal representation meant inhibitory neurons could not be tested. This is an underpowered / non-informative test for the precise inhibitory-neuron claim, not definitive evidence against it.
- No individual gene–cell-type result in the 75 prespecified tests survived multiple-testing correction. PDHA1 in inhibitory neurons had the largest nominal reduction (effect −0.418; q=0.829).
- In a four-cohort donor-level meta-analysis, PDHA1 was lower in AD in all four cohorts (pooled effect −0.457; 95% CI −0.712 to −0.201; I²=0%), but did not pass the gene-family adjusted threshold (q=0.086).
- A **post hoc** combined lipoylation/PDH/TCA transcriptional module was lower across the four cohorts (effect −0.245; 95% CI −0.363 to −0.127; I²=0%; q=0.035).
- The standalone OXPHOS and TCA modules did not reach the adjusted threshold and were heterogeneous (reported I² 75% and 78%).
- Copper-handling modules were essentially null. The authors explicitly state that the transcript data do not demonstrate cuproptosis.
- Similar or larger metabolic reductions were observed in Huntington's disease and vascular dementia; the AD-versus-Huntington and AD-versus-vascular-dementia contrasts were not significant.

**Interpretation for Neuro-Twin:** the data support a candidate disturbance around pyruvate oxidation / PDH / lipoylation / TCA at the cortical transcriptional level. In the large GSE33000 bulk cohort, the module remained lower after neuronal-content adjustment in the authors' independent recomputation (effect −0.113; 95% CI −0.213 to −0.013; p=0.026); OXPHOS also remained lower, while the TCA-only estimate did not reproduce as significant. In laser-captured GSE5281 neurons, OXPHOS and TCA were lower in all six regions, but the narrower PDH/lipoylation/TCA module reached significance in only one region. These controls make a pure glial-dilution explanation less likely, but do not show that this is the initiating lesion, do not demonstrate impaired mitochondrial flux or ATP production, and do not establish Alzheimer specificity. The independent single-nucleus replication is underpowered for the relevant neuronal compartment. A shared neurodegeneration-associated energy deficit remains at least as plausible as an Alzheimer-specific origin.

### 3.2 Parkinson: PFKFB2 and cell-state metabolic polarity

Sarhan (2026), DOI [10.1021/acsomega.6c07768](https://doi.org/10.1021/acsomega.6c07768), reanalysed GSE184950 and integrated public phosphoproteomic evidence.

Reported findings:

- The substantia-nigra atlas contained 390,360 nuclei after quality control, from 34 biological samples: 10 controls, 6 PD and 18 PDD.
- PFKFB2 was reduced across oligodendrocyte-lineage and neuronal compartments in PD; in PDD it was induced in microglia but reduced in astrocytic/glial-intermediate and neuronal compartments.
- PFKFB2-positive cell states were enriched for LRRK2–Rab trafficking, lysosomal/autophagy and glycolytic programmes.
- PFKFB2-negative states retained stronger mitochondrial/OXPHOS-associated transcriptional signatures.
- The paper positions PFKFB2 as a cell-state-resolved metabolic interface between trafficking and mitochondrial programmes, not as proof of primary mitochondrial failure.

**Interpretation for Neuro-Twin:** Parkinsonian substantia nigra shows a multicellular reorganisation of metabolic and endolysosomal programmes. This is inconsistent with a simplistic model in which every cell experiences the same uniform mitochondrial shutdown. GSE157783 provides an independent, smaller midbrain cohort (5 idiopathic PD, 6 controls; ~41.4k nuclei after one published QC pipeline) with processed UMI counts and cell annotations; it is the best first raw-data pilot for checking whether this broad pattern recurs. It raises a testable possibility that glycolytic adaptation, vesicle/lysosome handling and mitochondrial programmes become coupled differently across neurons, glia and disease stage. It does not tell us whether this reorganisation precedes dopaminergic cell loss.

## 4. First comparative judgment

The current public human evidence supports this provisional ranking of *what is justified*, not a ranking of biological causes:

1. **Shared bioenergetic / metabolic disturbance: plausible and supported as an associated process.** AD has a replicated PDH/lipoylation/TCA module signal; PD has cell-state-specific glycolytic/endolysosomal/OXPHOS transcriptional polarity.
2. **Metabolism as the universal first cause: not established.** Neither cross-sectional dataset identifies temporal precedence; transcript abundance is not pathway flux, respiration, ATP, NADH/NAD+ or redox state.
3. **Metabolism as an amplifier or vulnerability factor: currently more defensible than a universal initiator.** It can plausibly lower reserve and accelerate damage after other insults have begun.
4. **Disease-specific routing: plausible, not yet demonstrated by a joint analysis.** Candidate AD routing includes APP/Aβ/tau and cortical/hippocampal vulnerability; candidate PD routing includes alpha-synuclein, LRRK2/Rab trafficking, lysosome/autophagy and nigral dopaminergic vulnerability.
5. **Non-metabolic alternatives remain live.** Amyloid/clearance, tau, glial, vascular and mixed-threshold architectures must be run against the same observations and held-out cohorts.

A useful working architecture is therefore:

\`shared metabolic stress\` → \`reduced cellular resilience / redox-calcium strain\` → \`cell- and region-specific failure\` → disease-specific proteostasis and circuit pathology → persistent damage.

Feedback must be permitted: proteinopathy, glial activation and lysosomal failure can themselves impair mitochondria and energy handling. The causal direction cannot be hard-coded.

## 5. How the branch will test this instead of assuming it

### Prespecified biological modules

1. Glucose transport and glycolysis.
2. Pyruvate transport and PDH complex / lipoylation.
3. TCA cycle.
4. Electron transport chain and OXPHOS.
5. Mitochondrial biogenesis, dynamics and quality control / mitophagy.
6. Redox / NAD(H) / NADP(H), glutathione and calcium handling.
7. Endosome–lysosome / autophagic flux.
8. Proteostasis seeds and aggregation: AD Aβ/tau versus PD alpha-synuclein/LRRK2.
9. Glial and neurovascular support.

Genes are not to be treated as direct measurements of ATP, respiration, ROS, mitochondrial membrane potential or flux. Each measurement must carry assay, tissue, cell type, units, processing pipeline and uncertainty.

### Analysis design

- Use the biological donor, not each nucleus or array, as the independent unit.
- For single-nucleus cohorts, aggregate raw counts to donor × cell type before disease contrasts; never inflate sample size by counting nuclei as independent replicates.
- For repeated brain regions, use donor-aware hierarchical/mixed models or analyse each region separately if donor linkage is incomplete.
- Harmonise only comparable feature definitions and platforms; do not pool incompatible raw expression units.
- Test each gene set using a fixed, versioned gene list; account for multiple testing and report null results.
- Adjust or sensitivity-test age, sex, postmortem interval, RNA quality, brain region, cell composition, batch and disease stage where recorded.
- Replicate metabolic findings across independent cohorts, then compare whether the same signal is present in PD, AD and non-AD neurodegeneration controls.
- Reserve ADNI / PPMI for longitudinal temporal tests after authorized access: does metabolic change precede Aβ/tau or alpha-synuclein-related changes, neuronal injury and functional decline?
- Use genetic instruments and intervention/perturbation evidence as separate causal triangulation layers. Prediction or temporal association alone is not causal proof.

### Falsification conditions for metabolic-first

The strong metabolic-first hypothesis should be downgraded if:
- validated metabolic measures consistently change only after proteinopathy or neuronal injury;
- apparent signatures disappear after donor-level cell-composition and quality adjustment;
- early metabolic measures do not predict later damage beyond pathology and vascular/glial alternatives;
- the same intervention-level correction of the proposed metabolic lesion fails to alter downstream pathology/damage in a relevant perturbation system.

It should gain support if independent datasets show that a defined metabolic lesion is earlier than pathology and injury, robust to confounding, present within vulnerable cell populations, and experimentally perturbable in the predicted direction.

## 6. What has and has not been done

**Done in this audit:** source discovery; confirmation of public accession and sample metadata; extraction of quantitative results and limitations from two 2026 analyses of public human datasets; a first evidence-based comparative interpretation.

**Not done yet:** raw GEO matrices downloaded into Neuro-Twin; independent reanalysis; cell-type pseudobulk regeneration; gene-set scoring from raw counts; coefficient estimation; temporal precedence testing; integrated AD-versus-PD model fitting. No biological result is labelled as discovered by Neuro-Twin.

**Immediate next computational gate:** ingest GSE157783 first (the processed UMI matrix TAR is ~123.3 MB), validate the donor map and count provenance, and run prespecified donor-level pseudobulk energy-module tests. In parallel, use GSE174367's processed snRNA matrix (~261.3 MB) or GSE138852's public count/metadata resources for an AD cell-resolved pilot, after reconciling sample metadata. Then replicate with GSE184950 and GSE157827. The large raw packages for GSE184950 and GSE157827 are about 1.7 GB and 1.2 GB respectively. The current branch does not yet contain these count matrices or an independent raw-data result.

## References

- GSE184950: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184950
- GSE157827: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157827
- GSE33000: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000
- GSE5281: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5281
- GSE132903: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132903
- GSE48350: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48350
- GSE122063: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063
- GSE7621: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7621
- GSE8397: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8397
- Zhang et al. 2026, DOI: https://doi.org/10.1007/s12031-026-02598-7
- Sarhan 2026, DOI: https://doi.org/10.1021/acsomega.6c07768
- ADNI access: https://adni.loni.usc.edu/data-samples/adni-data/
- PPMI access: https://www.ppmi-info.org/access-data-specimens/download-data
- AMP-AD access policy: https://help.adknowledgeportal.org/apd/access-data
