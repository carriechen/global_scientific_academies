GSA Code Repository
===================

Overview
--------
This repository contains the code used to construct and analyze the Open Catalogue of Global Scientific Academies (GSA). The code is organized around the three evidence layers used in the study:

1. Layer 1: Organizational identity construction
2. Layer 2: Web structure mining and taxonomy construction
3. Layer 3: Academy-linked journal mapping and analysis
4. Figure: Figure and table generation scripts

The code is provided for reproducibility and research reuse. It is intended to work with the released GSA dataset, including the identity registry, web content artifacts, and linked journal files.

Repository structure
--------------------
code/
  layer1/
    Scripts for organizational identity construction and enrichment.

  layer2/
    Scripts for web structure mining, taxonomy construction, and website clustering.

  layer3/
    Scripts for ZDB journal retrieval, corporate-body mapping, hosted-journal analysis,
    genealogy reconstruction, and journal title lifecycle analysis.

  Figure/
    Scripts for reproducing manuscript figures and tables.


The stable field `acad_id` is the authoritative join key across identity, web, and journal layers.

Layer 1: Organizational identity construction
---------------------------------------------
The `layer1` folder contains scripts for constructing and enriching the organizational identity registry.

Typical tasks include:
- extracting candidate organizations from source lists
- traversing Wikipedia category pages
- assigning stable academy identifiers
- enriching external identifiers such as Wikidata, Wikipedia, DBpedia, ROR, VIAF, GND, LOC, and GRID
- completing geocoding and website language fields
- harvesting contact email information

Main output:
- `gsa_identity_registry.csv`
- `gsa_identity_registry.json`

These files form the organizational anchor for all downstream analyses.

Layer 2: Web structure mining and taxonomy construction
-------------------------------------------------------
The `layer2` folder contains scripts for extracting and analyzing academy website structure.

Typical tasks include:
- cleaning website URLs
- processing navigation-menu JSON files
- processing sitemap XML files
- extracting URL-path and menu-based hierarchical relations
- normalizing multilingual and heterogeneous labels
- constructing the web presence taxonomy
- generating website-level feature matrices
- calculating website-scale and URL-depth measures
- clustering academy websites by web-structural features
- calculating first-level category coverage by cluster

Main inputs:
- files in `gsa/web_content/navigation_menu/`
- files in `gsa/web_content/sitemaps/`
- the curated 110-site national academy subset used for web taxonomy analysis

Main outputs:
- web artifact summaries
- full web presence taxonomy
- website feature matrices
- clustering results
- category-coverage tables

These outputs support Figures 4–7 and the supplementary web taxonomy tables.

Layer 3: Academy-linked journal mapping and analysis
----------------------------------------------------
The `layer3` folder contains scripts for constructing and analyzing the academy-linked journal layer.

Typical tasks include:
- retrieving or processing ZDB journal records
- parsing ZDB corporate-body evidence
- constructing or validating `koeRef` to `acad_id` mappings
- filtering records to periodical journal title records
- calculating hosted-journal observability by academy group
- parsing journal publication spans
- reconstructing annual journal population dynamics
- processing DDC subject classifications
- reconstructing journal title genealogy families
- preparing Kaplan-Meier and RMST survival-analysis inputs

Main inputs:
- `gsa/linked_journal/gsa_linked_journal.csv`
- `gsa/linked_journal/gsa_zdb_koeRef_mapping.csv`
- `gsa/identity_registry/gsa_identity_registry.csv`

Main outputs:
- hosted-journal observability tables
- annual journal population tables
- DDC subject-time matrices
- genealogy component files
- journal title survival-analysis tables

These outputs support Figures 8–11 and Table 2.

Figure and table generation
---------------------------
The `Figure` folder contains scripts for reproducing manuscript figures.

Typical outputs include:
- Figure 1: Three-layer catalogue schema
- Figure 2: Entity-relationship diagram
- Figure 3: Founding periods by continent
- Figure 4: Website size and median URL depth
- Figure 5: Web presence taxonomy domain-size summary
- Figure 6: Website clustering by imitation and innovation
- Figure 7: First-level category coverage across clusters
- Figure 8: Fragmentation in academy-linked journal infrastructure
- Figure 9: Journal population dynamics
- Figure 10: Subject diversification of academy-linked journals
- Figure 11: Largest journal lineage family

Figure scripts read from curated or derived files. They should not modify the underlying data files.

Recommended workflow
--------------------
A typical reproduction workflow is:

1. Prepare the public GSA dataset using the folder structure described above.
2. Run or inspect `layer1` scripts to reproduce identity registry construction and enrichment.
3. Run or inspect `layer2` scripts to reproduce web artifact processing, taxonomy construction, clustering, and category coverage.
4. Run or inspect `layer3` scripts to reproduce ZDB journal linkage, hosted-journal observability, genealogy, and survival-analysis inputs.
5. Run scripts in `Figure` to regenerate manuscript figures and tables.

Some steps involve curated intermediate files based on manual validation. These files are part of the reproducible research object and should not be replaced by live web searches or uncontrolled registry queries unless a new dataset version is being created.

Reproducibility notes
---------------------
The project integrates dynamic external sources, including official websites, open registries, and ZDB records. Exact regeneration from live sources may not reproduce the archived Version 1 dataset because websites, registry identifiers, and bibliographic metadata may change over time.

For reproducing the published analyses, use the archived Version 1 release files. Manual validation decisions, including academy inclusion, identifier matching, web taxonomy normalization, and ZDB corporate-body mapping, are preserved in curated files and should be treated as versioned research objects.

Dependencies
------------
The code was developed in Python. Commonly used packages include:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- networkx
- beautifulsoup4
- lxml
- openpyxl

Additional dependencies may be required by individual scripts. See script headers or the environment file, if provided.

License
-------
Recommended conservative licensing:

- Code: MIT License
- Data: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)


Users should cite the associated paper and dataset when reusing the data, code, or derived outputs.

Citation
--------
Suggested citation format:

Chen, X., & Wang, X. (2024). Profiling Global Scientific Academies. In Proceedings of the 24th ACM/IEEE Joint Conference on Digital Libraries. ACM. https://doi.org/10.1145/3677389.3702582
Chen, X., & Wang, X. (2025). Web Mining the Online Presence of Global Scientific Academies. In 20th International Conference on Scientometrics & Informetrics. Institute for Informatics and Automation Problems of NAS RA. https://doi.org/10.51408/issi2025_008


Contact
-------
For questions about the code or dataset, contact the corresponding authors listed in the associated manuscript.
