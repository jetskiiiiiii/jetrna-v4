jetRNA v4

Ideas: 
- create bigger dataset
- improve CNN architecture
- use GNNs for transfer learning to predict pseudoknots/base-stacking
- sort through known shapes for specific nucleotide patterns to give general shape
    - search through database of structures which guides training process
    -  At inference time we provide the model the top 4 templates, sorted by the expected number of correctly
aligned residues (the “sum_probs” feature output by HHSearch). At training time we first restrict the
available templates to up to 20 with the highest “sum_probs”. Then we choose random k templates out
of this restricted set of n templates, where k = min(Uniform[0, n], 4). This has the effect of showing
the network potentially bad templates, or no templates at all, during training so the network cannot rely
on just copying the template.
- create a structure module that maps 2d to 3d

To-do:
- (DONE) get data from CT
- (DONE) create pair/single representation

Notes:
- Order of bases: C, G, A, U

- Single represenation
    - one hot
    - di nucleotide frequencies (identity of consecutive bases)
    - local content (proportion of g/c per window size centered around base)
        - g/c
            - G and C bases form three hydrogen bonds when they pair (G-C), while A and U/T bases form only two hydrogen bonds (A-U/T). Consequently, a region of the sequence with a higher local G/C content is generally more stable + rigid.
            - Regions with high local G/C content are strong predictors of stable helical (stem) regions in both DNA and RNA secondary structures.
        - a/u
            - Regions with low local G/C content (high A/U or A/T content) often correspond to less stable structural elements like loops or unstructured regions that are more prone to flexibility and tertiary interactions.
    - potential inverted
        - 5' -- X -- SPACER -- X' -- 3'
        - min lengths: 4, 5, 6

- files:
    - bprna, archiveII, rnastralign
        - CT - sequence and pairings (no pseudoknots)
    - bprna
        - ST - sequence, pairings, structural elements (with pseudoknots)
        - DBN

- Thoughts on how to think (and talk) about RNA structure:
https://www.pnas.org/doi/10.1073/pnas.2112677119
    - Stacking as a Driver for RNA Structure
        - negatively charged ribose–phosphate chain—which makes up two-thirds of the mass of a nucleotide—creates steric and electrostatic constraints on the backbone conformation
        - Bases, which account for the remaining third of mass, comprise planar aromatic rings decorated with partially charged hydrogen-bond donors and acceptors
        - every nucleotide in an RNA chain can favorably interact with every other nucleotide is critical for thinking about RNA structure and how it differs from protein structure.
        - the helical structure of RNA (and DNA) is not induced by Watson–Crick pairing but by base stacking
    - “Inherently Structured” Does Not Mean “Static”
        - conformation of an RNA is dictated by a thermodynamic landscape that depends on the sequence of the RNA and the conditions
    - RNA Is a Compact Molecule 
        - Compactness and long-distance pairing, combined with the modularity of RNA and the increase of RNA size over time by accretion, result in the 5′ and 3′ ends of any natural RNA being in relatively close spatial proximity
    - Watson–Crick Pairing Is Important but a Bit Overrated
        - in most cases of folded RNA three-dimensional (3D) structures, non-Watson–Crick pairs are critical for creating the tertiary interactions that stabilize the functional conformation
        - GC content is not a reliable predictor of “highly structured” RNAs, unless one knows a priori that most of the nucleotides are involved in Watson–Crick pairs
    - Non-Watson–Crick Pairing Is Very Much Underrated
        - Watson–Crick paired helical regions generally act as spacers with regular geometrical properties essential to position structural elements stabilized by non-Watson–Crick pairs.
        - While it is true that isolated non-Watson–Crick pairs are destabilizing within A-form helices, it is equally true that non-Watson–Crick pairs form favorable, specific, and necessary interactions in specific structural contexts
    - perhaps the issue is in Depicting RNA

- Task logger:
    10/30
        - finished creating single/pair representation
    10/31
        - finished CT file parsing (to extract sequenece, pairings)
        - created script to move CT files
        - put together sequences from RNAStrAlign and ArchiveII as parquet
