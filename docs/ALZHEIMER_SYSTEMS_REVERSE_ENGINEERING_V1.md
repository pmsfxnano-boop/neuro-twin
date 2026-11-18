# NEURO-TWIN — Alzheimer disease systems reverse engineering v1.0

**Research status:** evidence synthesis and computational decomposition  
**Disease scope:** Alzheimer disease only  
**Prepared:** 18 November 2026  
**Model status:** research hypothesis architecture; not a validated causal model, clinical diagnostic system, or treatment recommender

## 1. Mission and scientific boundary

The Neuro-Twin objective is not merely to classify Alzheimer disease, reproduce clinical scores, or forecast cognitive decline. The research objective is to reconstruct the biological mechanisms through which Alzheimer pathology can originate, become self-amplifying, accumulate tissue damage, and express as neuronal-network and cognitive failure; then identify mechanistically credible points and time windows at which those trajectories might be interrupted.

This is a reverse-engineering problem across scales:

1. Start from clinical and anatomical phenotypes and trace backward through circuits, cells, organelles, molecular pathways, genetic susceptibility and life-course exposures.
2. Start from candidate upstream perturbations and trace forward through pathology, feedback loops, tissue damage and functional consequences.
3. Reconcile both directions using human genetics, longitudinal human observations, neuropathology, imaging, intervention evidence and controlled perturbation experiments.

The current evidence does **not** establish one universal initiating event for all Alzheimer disease. Rare autosomal-dominant disease has strong genetic causes involving APP, PSEN1 or PSEN2. Most cases are late-onset and sporadic, with heterogeneous combinations of age-related vulnerability, genetic susceptibility, altered protein handling, immune biology, vascular/metabolic contributors and co-pathology. Neuro-Twin must therefore compare competing, stage-specific causal hypotheses rather than assume a single sequence.

Alzheimer disease is a biological process, not a synonym for dementia. Amyloid and tau pathology can precede symptoms by many years; clinical impairment depends additionally on regional neuronal injury, vascular lesions, co-pathologies, baseline reserve and compensation. Biological stage and clinical stage must remain separate axes.

## 2. What is known about origins — and what is not

### 2.1 Autosomal-dominant Alzheimer disease

Pathogenic variants in **APP, PSEN1 and PSEN2** cause rare autosomal-dominant Alzheimer disease (ADAD), generally with high penetrance. This is the clearest human genetic route into altered amyloid-beta biology. APP encodes the amyloid precursor protein; presenilins are components of the gamma-secretase complex. These mutations can alter peptide production and the distribution of amyloid-beta species.

This route is exceptionally informative for causal discovery, but it represents a small minority of Alzheimer dementia. It cannot simply be assumed to explain all sporadic late-onset disease.

### 2.2 Down syndrome and APP dosage

Chromosome 21 contains APP. Increased APP gene dosage in Down syndrome is associated with a high lifetime probability of Alzheimer neuropathology. This is another genetic dosage route into amyloid biology, but its developmental and systemic context is distinct and should not be merged uncritically with either ADAD or typical late-onset disease.

### 2.3 Sporadic late-onset Alzheimer disease

Most cases are not explained by a single highly penetrant mutation. The strongest common genetic risk factor is **APOE epsilon-4**, but it is a probabilistic risk allele, not a deterministic diagnosis. Other risk loci implicate lipid transport, endosomal trafficking, innate immunity, microglial function and protein clearance, including pathways represented by genes such as TREM2, SORL1, ABCA7, BIN1, PICALM, CLU and related loci.

The engineering interpretation is that susceptibility may alter several coupled capacities:

- production and clearance of amyloid-beta;
- lipid transport and membrane/organelle homeostasis;
- endosomal sorting and lysosomal delivery;
- innate immune sensing and phagocytosis;
- repair, resilience and response to aging-related stress.

A genetic association is not automatically a molecular mechanism. Every locus-to-mechanism link needs variant-level evidence, relevant tissue/cell context, colocalization or equivalent fine-mapping where appropriate, and functional validation.

### 2.4 Aging, vascular and life-course modifiers

Aging changes proteostasis, mitochondrial quality control, vascular integrity, immune regulation and repair capacity. Midlife hypertension, diabetes, obesity and smoking are associated with later vascular brain injury and cognitive risk. These exposures may influence Alzheimer-specific mechanisms, add independent vascular injury, or both.

The engine must distinguish:
- a factor that initiates Alzheimer neuropathologic change;
- a factor that increases susceptibility to it;
- a factor that accelerates an existing pathological process;
- a co-pathology that adds damage independently;
- a factor that changes clinical expression without changing the underlying pathology.

Sleep/circadian disruption, physical inactivity, sensory impairment, infection/inflammatory exposures, endocrine factors and microbiome-related mechanisms belong in the wider research map, but their causal maturity varies. They must not be promoted to primary causal drivers merely because associations have been reported.

## 3. Multiscale decomposition of Alzheimer pathobiology

The following modules are not independent boxes. They form an interacting network with reciprocal feedback, cell-type specificity, regional vulnerability and changing influence across disease stages.

### 3.1 APP processing and amyloid-beta production

APP can be processed through non-amyloidogenic and amyloidogenic routes. In the amyloidogenic route, beta-secretase cleavage generates a C-terminal fragment that is subsequently cleaved by gamma-secretase, producing amyloid-beta peptides, including A-beta-40 and A-beta-42.

A-beta-42 is more aggregation-prone. The disease-relevant system is not captured by total amyloid concentration alone. It includes:

- APP expression, trafficking and cleavage location;
- secretase activity and peptide-species distribution;
- soluble monomers, oligomers and higher-order assemblies;
- fibrillar parenchymal plaques;
- vascular amyloid and cerebral amyloid angiopathy (CAA);
- degradation, transport and clearance through cellular and vascular routes.

**Critical distinction:** soluble A-beta assemblies are candidate synaptotoxic species; fibrillar plaques are a different physical and anatomical pool. A plaque signal is not a direct measure of the concentration of toxic soluble oligomers at synapses.

### 3.2 Amyloid clearance and trafficking

Brain amyloid burden reflects both production and clearance. Candidate clearance routes include enzymatic degradation, receptor-mediated transport across the blood-brain barrier, cellular uptake and endolysosomal processing, and perivascular/interstitial-fluid drainage.

Relevant candidate systems include LRP1-related transport, neprilysin and insulin-degrading enzyme activity, APOE-dependent transport, endosomal sorting, lysosomal function, vascular basement membrane, pericytes and perivascular drainage.

The distinction between production excess and clearance failure is essential. Familial APP/presenilin disease can involve altered production/species generation; late-onset disease may prominently involve impaired clearance and cellular handling. They should not be represented by one undifferentiated amyloid input.

### 3.3 Tau misprocessing, aggregation and propagation

Tau normally supports microtubule-associated functions and axonal transport. In Alzheimer disease, tau can undergo abnormal phosphorylation, conformational change, mislocalization and aggregation. Tau pathology is not one scalar event:

- phosphorylation at different sites has different biological and biomarker meanings;
- soluble abnormal tau and aggregated tau are not equivalent;
- intracellular tangles, neuritic tau and extracellular seeds are distinct compartments;
- tau pathology has regional topography and can propagate through connected neural systems;
- cellular vulnerability and the local microenvironment modify the consequences.

Tau burden and its regional distribution often track neuronal injury and clinical decline more closely than plaque burden. That does not prove that tau is the first initiating event in every individual. A central Neuro-Twin question is whether amyloid acts upstream to facilitate tau pathology, whether tau becomes partly self-propagating, and when the downstream process becomes less dependent on the initial amyloid perturbation.

### 3.4 Endosomal, lysosomal and proteostasis systems

Protein homeostasis is a network, not a single “protein clearance” variable. It includes:

- endosomal uptake, sorting, recycling and trafficking;
- retromer-dependent cargo handling;
- chaperone-mediated folding and refolding;
- ubiquitin tagging and proteasomal degradation;
- autophagosome formation and maturation;
- lysosomal acidification, enzyme activity and cargo degradation;
- transport of damaged proteins and organelles;
- proteostasis signaling and cellular stress responses.

A key modeling issue is **flux**. A high autophagosome or lysosomal-marker level may indicate increased production, blocked degradation, or both. Static abundance must not be treated as proof of effective clearance. Neuro-Twin must distinguish pathway activity from pathway throughput.

Endosomal and lysosomal disruption can also affect APP trafficking, receptor turnover, lipid handling, mitochondrial quality control and inflammatory signaling. This makes proteostasis a potential upstream susceptibility module and a downstream amplifier.

### 3.5 Neuroimmune and glial response

Microglia and astrocytes are context-dependent regulators, not simply “good” or “bad” inflammatory cells.

Microglia participate in surveillance, amyloid uptake, debris clearance, synaptic remodeling and inflammatory signaling. TREM2/APOE-associated programs and other immune pathways can change these functions. Persistent or maladaptive states may contribute to cytokine signaling, inflammasome activation, complement activity and altered phagocytosis.

Astrocytes regulate metabolic substrate delivery, lipid transport, glutamate handling, extracellular ion balance, synaptic support and neurovascular coupling. Reactive states can be protective, compensatory or harmful depending on stage and context.

Candidate mechanisms include:
- altered microglial sensing and phagocytic clearance;
- complement tagging and excessive synaptic elimination;
- inflammasome/cytokine activity;
- astrocytic lipid and metabolic support changes;
- failure to resolve inflammation;
- peripheral-to-central immune signaling.

The I coordinate must not imply that every inflammatory response is toxic. A single activation magnitude may be inadequate; future model versions may need separate phagocytic/clearance, inflammatory, and resolution state variables.

### 3.6 Mitochondria, bioenergetics, lipid metabolism and redox balance

Neurons have high and sustained energy requirements. Candidate abnormalities include:

- altered glucose uptake and utilization;
- impaired oxidative phosphorylation and ATP availability;
- changes in mitochondrial fission, fusion, transport and mitophagy;
- calcium handling and mitochondria–endoplasmic-reticulum contact changes;
- altered NAD+/NADH balance and redox capacity;
- glutathione and antioxidant-system disruption;
- lipid transport, cholesterol handling and membrane remodeling;
- lipid-droplet accumulation and altered fatty-acid utilization;
- oxidative and nitrosative stress.

These processes interact with APP processing, tau, glial states, synaptic transmission and lysosomal clearance. The relationship can be bidirectional: metabolic failure may amplify proteinopathy, while proteinopathy and inflammation can further damage mitochondrial and metabolic systems.

FDG-PET measures regional glucose utilization; it is not a direct measurement of mitochondrial ATP production. Likewise, a single oxidative-stress marker cannot define an entire redox system.

### 3.7 Neurovascular unit, blood-brain barrier and drainage

The neurovascular unit includes endothelial cells, pericytes, astrocytic end-feet, vascular smooth muscle, basement membrane and associated neural elements. Its functions include perfusion, barrier integrity, neurovascular coupling and exchange/clearance.

Candidate mechanisms:
- endothelial dysfunction and barrier leakage;
- pericyte injury or loss;
- impaired cerebral blood flow and autoregulation;
- capillary stalling and altered oxygen/substrate delivery;
- cerebral small-vessel disease and microinfarcts;
- CAA and vessel-wall injury;
- altered perivascular/interstitial drainage and glymphatic-associated clearance.

Vascular dysfunction may precede detectable Alzheimer biomarker abnormalities in some cohorts and can contribute independently to cognitive decline. It can also interact with amyloid clearance and CAA. MRI white-matter hyperintensities are not a specific measurement of every microvascular mechanism; much capillary-level pathology remains difficult to observe in vivo.

### 3.8 Synaptic, neuronal and circuit failure

The clinically relevant output is not reducible to plaque count. Candidate downstream mechanisms include:

- loss of synaptic density and synaptic plasticity;
- impaired long-term potentiation and memory encoding;
- glutamatergic imbalance and excitotoxic stress;
- altered neuronal excitability and network synchrony;
- cholinergic-system dysfunction;
- axonal transport failure and neuritic dystrophy;
- regional neuronal vulnerability;
- loss of neuronal and network connectivity;
- neuronal death and regional atrophy.

Synaptic dysfunction can precede overt neuronal loss. Neuronal and circuit integrity (N) must remain separate from realized cognitive/functional capacity (Q): compensation, reserve, treatment, sleep, mood, sensory function and co-pathologies can change performance without proportionate changes in structural injury.

### 3.9 Co-pathology and mixed disease

A person with Alzheimer neuropathologic change may also have:
- cerebral small-vessel disease, infarcts or microbleeds;
- cerebral amyloid angiopathy;
- Lewy-body / alpha-synuclein pathology;
- TDP-43 pathology, including LATE;
- hippocampal sclerosis;
- other age-related proteinopathies.

These can add injury, change regional vulnerability, alter biomarkers and modify treatment response. Neuro-Twin must not label all cognitive decline in an amyloid-positive person as amyloid/tau-mediated Alzheimer progression.

### 3.10 Systemic modifiers and regional susceptibility

Age, genetic background, sex, vascular/metabolic health, life-course exposures and systemic disease can modify the state of the brain. Regional cell populations differ in connectivity, energetic demand, protein expression, clearance capacity and vulnerability. The entorhinal/hippocampal memory system and association cortices are important but not the only affected regions.

A future model must use region-aware states or region-specific parameters where the data identify them. A global scalar may hide opposing regional trajectories.

## 4. Candidate feedback loops to test

These are mechanistic hypotheses to encode and test, not universal causal facts.

1. **Amyloid–immune loop:** A-beta species alter glial sensing and inflammatory/phagocytic state; glial responses may alter A-beta clearance, aggregation and tissue injury.
2. **Amyloid–tau loop:** A-beta-associated stress can facilitate tau misprocessing; tau-related neuronal stress can in turn alter clearance and proteostasis.
3. **Proteostasis–mitochondria loop:** impaired lysosomal/autophagic flux reduces organelle quality control; mitochondrial dysfunction increases oxidative and proteotoxic load.
4. **Metabolic–synaptic loop:** reduced energy and altered lipid/redox handling impair synaptic maintenance; dysfunctional circuits increase energetic stress.
5. **Vascular–amyloid loop:** vascular/pericyte dysfunction may impair perfusion and perivascular clearance; amyloid deposition in vessel walls can worsen vascular integrity.
6. **Damage–clearance loop:** loss of cellular and tissue integrity increases debris and clearance demand, which can further burden glial and lysosomal systems.
7. **Reserve–phenotype decoupling:** reserve and compensation can delay clinical expression despite ongoing biological damage; when compensatory capacity is exhausted, clinical decline may become more visible.

The map in backend/neuro_twin/network/alzheimer_mechanism_map_v1.json contains an initial 28-node / 30-edge evidence-graded inventory. Its directed edges are candidate mechanisms, not a validated directed acyclic graph. It has no fitted edge coefficients and cannot be executed as a causal simulation without additional evidence, parameterization and validation.

## 5. Biological time: sequence, stages and uncertainty

The 2024 Alzheimer's Association criteria distinguish biological Alzheimer pathology from clinical impairment. Core 1 biomarkers can establish the presence of Alzheimer biological change; later-changing Core 2 biomarkers, including tau PET and additional fluid biomarkers, add staging and prognostic information. Neurodegeneration markers do not follow one perfectly stereotyped order in every person, and co-pathologies can cause substantial injury.

DIAN provides unusually deep longitudinal information for rare autosomal-dominant disease. In this population, amyloid PET and CSF A-beta changes can begin many years before symptoms, followed by changing phosphorylated-tau markers, neurodegeneration markers, hypometabolism, atrophy and later cognitive decline. This sequence is highly informative for ADAD but must not be copied as a fixed clock for all sporadic disease.

Neuro-Twin therefore needs:
- biological time and chronological time as distinct variables;
- stage-dependent and individual-specific transition rates;
- irregular visit-time handling;
- explicit missingness and assay versioning;
- prospective point-in-time estimates separated from retrospective smoothing;
- uncertainty about biomarker ordering and latent-state observability;
- subject-level and external cohort validation.

No universal “years until dementia” clock should be inferred from the ADAD sequence.

## 6. The new Damage Accumulation module

### 6.1 Why it is separate

Damage accumulation is not the same thing as:
- P: current disease-process / progression burden;
- N: currently preserved neuronal and circuit integrity;
- Q: realized functional capacity;
- I, M or Pr: current inflammatory, metabolic or proteostasis states.

Two people can have similar current stress or biomarker levels but very different accumulated tissue loss because their exposure histories, repair capacity, regional vulnerability and co-pathologies differ. Conversely, a person can maintain function temporarily despite substantial damage because reserve and compensation are still available.

### 6.2 Persistent damage compartments

The first damage module tracks four distinct compartments:

1. synaptic_structural — persistent synaptic/neuritic structural loss;
2. axonal_neuritic — persistent axonal and neuritic injury;
3. neuronal_somatic — accumulated neuronal cell-body loss;
4. neurovascular — accumulated microvascular and neurovascular structural injury.

These are research constructs, not direct biomarkers. A future region-aware extension may split each compartment by brain region and cell type.

### 6.3 Mathematical contract

Let u(t) be explicitly mapped damaging exposures and r(t) explicitly mapped repair/resilience capacities, each normalized by a documented upstream model. For damage compartment k:

\[
\dot D_k(t)=s_k\left[\mathbf{w}_k^\top \mathbf{u}(t)
-\mathbf{v}_k^\top \mathbf{r}(t)\right]_+,
\qquad
D_k(t)=D_k(t_0)+\int_{t_0}^{t}\dot D_k(\tau)\,d\tau
\]

where [z]+ = max(z, 0), weights are non-negative and explicitly supplied, and s_k is a compartment-specific rate scale. The first implementation uses piecewise-linear interpolation of supplied visit-level rates and trapezoidal integration for irregular visits.

Interpretation:
- persistent damage is non-decreasing in this first model;
- repair capacity reduces the rate of new persistent damage;
- repair does not erase prior neuronal cell loss or prior structural injury;
- transient/reversible dysfunction is not added to D unless a validated conversion model shows persistent injury;
- all input normalization, weights, rate scales and time units must be explicit and traceable.

This monotone state is deliberately conservative. Synaptic remodeling and some tissue processes can recover; those processes belong in separate reversible injury/plasticity states. The current module represents persistent accumulated burden, not all forms of biological injury.

### 6.4 Important non-equivalences

- NfL is a marker of aspects of axonal injury/turnover, not a direct measurement of total irreversible damage.
- MRI atrophy is a regional structural proxy, not a cell count.
- Synaptic-density imaging, if available, measures a different substrate from cognitive scores.
- Cognitive performance can improve or fluctuate without reversal of accumulated neuronal cell loss.
- A global D score must not be produced by averaging compartments or regions without a validated aggregation model.

### 6.5 First software implementation

backend/neuro_twin/network/damage.py implements:
- a disease-scoped model for Alzheimer or Parkinson;
- four explicit persistent-damage compartments;
- explicit damaging-exposure and repair-capacity input names;
- non-negative, caller-supplied mapping matrices and rate scales;
- normalized input checks;
- irregular-time trapezoidal integration;
- deterministic configuration hashing;
- a monotone cumulative output.

It is a sidecar module, not yet a coupled seven-state joint model. The existing six-state model remains unchanged. The current implementation deliberately does not invent mappings from P/I/M/N/Q/Pr or from raw biomarkers to damage inputs. Those mappings require measurement definitions, evidence, cohort data and identifiability checks.

## 7. Intervention logic: where interruption might be possible

Neuro-Twin must not simply ask which target is abnormal. It must ask whether changing that target at a specified stage changes the downstream damage trajectory, through which mediators, with what uncertainty and for which biological subtype.

Candidate intervention layers include:

| Layer | Candidate leverage point | Main scientific question |
|---|---|---|
| Genetic / susceptibility | APOE, endosomal and immune-risk pathways | Can susceptibility be modified safely or compensated before pathology accelerates? |
| A-beta production / clearance | APP processing, aggregation, antibody-mediated removal, transport/clearance | Which A-beta species and compartments must change, and when? |
| Tau | phosphorylation, aggregation, seeding and spread | Can tau propagation be interrupted after amyloid is already present? |
| Proteostasis | endosomal sorting, lysosomal flux, autophagy and protein quality control | Can clearance capacity be restored without harmful off-target effects? |
| Neuroimmune | microglial phagocytosis, inflammatory state, complement and resolution | Can harmful states be reduced while preserving protective clearance and repair? |
| Metabolic / mitochondrial | energy substrate use, lipid handling, mitochondrial quality, redox and calcium | Are these upstream drivers, amplifiers, or consequences in a given subtype? |
| Neurovascular | perfusion, pericytes, barrier function and drainage | Can vascular injury and impaired clearance be slowed before persistent tissue injury? |
| Synaptic / neuronal | excitability, plasticity and trophic support | Can network function be preserved while upstream pathology is addressed? |
| Damage / resilience | reduce damaging flux, support repair and preserve reserve | Can the rate of new persistent damage be reduced before compensation is exhausted? |

Clinical anti-amyloid interventions provide evidence that modifying one pathological pathway can change clinical trajectory in selected early symptomatic populations, but they do not establish a cure or full reversal. Continued decline after amyloid reduction highlights the need to model downstream tau, inflammation, synaptic injury, vascular contributions and accumulated damage. Treatment effects, adverse effects, eligibility and regional authorization belong to clinical evidence systems; Neuro-Twin must not become a patient-level treatment recommender.

The 2026 drug-development pipeline spans many mechanistic categories. The number of drugs or trials in a category measures research activity, not therapeutic success or causal proof.

## 8. Causal discovery and model-engineering rules

### 8.1 Evidence hierarchy and triangulation

For every proposed mechanism or edge, record:
- evidence class: human genetics, longitudinal human data, human neuropathology, human imaging, human biomarker, clinical intervention, in-vitro perturbation, animal model, or review synthesis;
- population, disease subtype, age range and cell/tissue context;
- region and biological scale;
- time direction and stage;
- assay, preprocessing, cohort and source version;
- effect direction, uncertainty and possible confounders;
- contradictory or null evidence;
- whether the link is association, mediation, intervention-supported, genetically supported or still hypothesis-only.

A review citation is useful for organizing hypotheses; it does not turn an edge into causal truth.

### 8.2 Required analytical methods

Neuro-Twin's causal research layer should combine, rather than rely on a single method:

1. **Human genetics:** GWAS fine-mapping, rare-variant analysis, brain-relevant eQTL/pQTL, colocalization and Mendelian randomization with pleiotropy/sensitivity checks.
2. **Longitudinal human evidence:** temporal ordering, mixed-effects/state-space models, cross-lagged hypotheses with appropriate assumptions, within-person trajectory analysis and target-trial-compatible analyses where possible.
3. **Multi-omics:** transcriptomic, proteomic, metabolomic, lipidomic, epigenomic and spatial/single-cell data with cell-type and tissue context.
4. **Causal graphs:** competing DAGs and dynamic Bayesian networks, with explicit latent confounding, selection, measurement error and feedback limitations.
5. **Perturbation:** CRISPR, isogenic human iPSC-derived cells, organoid/co-culture systems and animal models chosen for the mechanism being tested.
6. **Clinical intervention:** randomized perturbation evidence and mediation analyses, including negative trials and downstream persistence after target engagement.
7. **Independent replication:** different cohorts, platforms, centers and populations.

The 2026 review of causal analyses in Alzheimer molecular data specifically warns that observational associations are not sufficient to identify causal or treatable targets, and that tissue context, estimand and method assumptions can produce apparently conflicting results.

### 8.3 Causal status is not coefficient status

A mechanism map edge can be:
- genetically supported;
- intervention supported;
- supported by convergent but non-definitive evidence;
- association only;
- hypothesis only.

Even a strongly supported biological relationship does not supply a numerical coefficient for the Neuro-Twin dynamics. Coefficients require defined units, parameter identifiability, calibration data, uncertainty estimates and out-of-sample testing.

## 9. How this decomposition changes the Neuro-Twin engine

### 9.1 Current foundation

The six-domain package provides P/I/M/N/Q/Pr, explicit observations, continuous-time dynamics, filtering/smoothing and mathematical identifiability diagnostics. It is currently an affine hypothesis scaffold, not a validated biological model. The new Damage Accumulation module is separate.

### 9.2 Required architecture evolution

The next motor versions should move toward a hierarchical, disease-specific, region-aware state-space architecture:

- **Layer 0 — evidence graph:** typed nodes/edges, evidence maturity, source provenance and competing causal hypotheses;
- **Layer 1 — biological mechanisms:** APP/A-beta, tau, endosomal-lysosomal flux, glial states, metabolism/redox, neurovascular and synaptic mechanisms;
- **Layer 2 — latent domains:** P/I/M/N/Q/Pr plus persistent damage D, without double-counting constructs;
- **Layer 3 — regional/cellular structure:** region-specific vulnerability and cell-type contributions where identifiable;
- **Layer 4 — observation operators:** modality- and assay-specific likelihoods; no automatic mapping from biomarker name to latent state;
- **Layer 5 — dynamic inference:** nonlinear/stochastic dynamics, regime changes and feedback where evidence and sample size support them;
- **Layer 6 — intervention and counterfactual simulator:** explicit target perturbations, timing, mediator pathways, uncertainty and failure/abstention;
- **Layer 7 — validation:** point-in-time prediction, temporal holdouts, subject-level splits, external cohorts, posterior predictive checks, calibration and parameter recovery.

The affine six-state model should remain a transparent baseline. A more complex nonlinear model must earn its complexity through identifiability and predictive or mechanistic gain.

### 9.3 Near-term implementation order

1. Validate the new mechanism-map JSON and the damage module in CI.
2. Add damage-specific observation/measurement definitions and provenance contracts.
3. Build source adapters for actual ADNI, DIAN and other authorized longitudinal data; do not invent source aliases.
4. Quantify which mechanisms and damage compartments are observable from each cohort and visit schedule.
5. Compare rival Alzheimer-origin graphs using genetics and longitudinal human evidence.
6. Fit only identifiable parameters, with hierarchical shrinkage and uncertainty.
7. Test the ability to recover known synthetic perturbations before any biological interpretation.
8. Evaluate temporal OOS prediction and causal/interventional hypotheses in independent data.
9. Only then couple D to the six-state dynamics and permit counterfactual interruption experiments.

## 10. Falsifiable research questions for Neuro-Twin

The project should generate explicit, refutable hypotheses, for example:

- In which Alzheimer subgroups does impaired clearance explain amyloid accumulation better than increased production?
- Does endosomal/lysosomal dysfunction precede measurable tau acceleration, or does it mainly amplify already established pathology?
- Which human longitudinal signals support an amyloid-to-tau transition, and in which groups does the sequence differ?
- Does neurovascular dysfunction precede Alzheimer Core 1 biomarker positivity in a reproducible subset, or does it mainly add independent injury?
- Which glial signatures are associated with effective clearance versus persistent inflammatory damage?
- At what point does tau-associated synaptic injury become partly independent of further amyloid reduction?
- Can damage-accumulation trajectories be estimated independently of current N, P and Q states?
- Does reducing a proposed upstream driver lower future D accumulation in held-out subjects or perturbation systems?
- Which mechanisms are common across individuals and which define distinct biological subtypes?

A hypothesis is not a discovery until it survives preregistered tests, alternative explanations, negative controls and independent replication.

## 11. Key sources

1. Jack et al. (2024), *Revised criteria for diagnosis and staging of Alzheimer's disease*, Alzheimer's & Dementia. DOI: [10.1002/alz.13859](https://doi.org/10.1002/alz.13859).
2. Hein et al. (2025), *Toward a Unified Framework in Molecular Neurobiology of Alzheimer's Disease: Revisiting the Pathophysiological Hypotheses*. DOI: [10.1007/s12035-025-05602-0](https://doi.org/10.1007/s12035-025-05602-0).
3. Hein et al. (2025), *An integrated view of the relationships between amyloid, tau, and inflammatory pathophysiology in Alzheimer's disease*. DOI: [10.1002/alz.70404](https://doi.org/10.1002/alz.70404).
4. Bice et al. (2026), *Systems biology of Alzheimer's Disease: a scoping review of key pathways and mechanisms*. DOI: [10.1186/s13024-026-00934-4](https://doi.org/10.1186/s13024-026-00934-4).
5. Maioli et al. (2025), *Bioenergetics and lipid metabolism in Alzheimer's disease: From cell biology to systemic health*. DOI: [10.1111/joim.70036](https://doi.org/10.1111/joim.70036).
6. Divecha et al. (2025), *The microcirculation, the blood-brain barrier, and the neurovascular unit in health and Alzheimer disease: The aberrant pericyte is a central player*. DOI: [10.1016/j.pharmr.2025.100052](https://doi.org/10.1016/j.pharmr.2025.100052).
7. Daniels et al. (2026), *15 years of longitudinal genetic, clinical, cognitive, imaging, and biochemical measures in DIAN*. npj Dementia. [Article](https://www.nature.com/articles/s44400-025-00047-7).
8. Tansey et al. (2026), *Alzheimer's disease heterogeneity and co-pathologies: Defining novel targets, biomarkers, modalities, and research methodologies*. DOI: [10.1002/trc2.70287](https://doi.org/10.1002/trc2.70287).
9. Cummings et al. (2026), *Alzheimer's disease drug development pipeline: 2026*. DOI: [10.1002/trc2.70251](https://doi.org/10.1002/trc2.70251).
10. Loo & Glaab (2026), *Causal analysis of Alzheimer's disease molecular data*. DOI: [10.1186/s13195-026-02235-w](https://doi.org/10.1186/s13195-026-02235-w).
11. Zhang et al. (2025), *Pathological mechanisms and treatment progression of Alzheimer's disease*. DOI: [10.1186/s40001-025-02886-9](https://doi.org/10.1186/s40001-025-02886-9).

These sources support the decomposition and the design of candidate hypotheses. They do not validate Neuro-Twin's latent states, edge directions, coefficients, accumulated-damage estimates or intervention predictions.
