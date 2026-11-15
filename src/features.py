"""
Turn an siRNA guide sequence into numbers a model can learn from.

Three kinds of feature, all standard in the siRNA design literature:

1. Composition    - which bases appear, and in which positions.
2. Thermodynamics - RNA nearest-neighbour free energies (Xia et al. 1998). The key
                    quantity is *asymmetry*: a guide strand whose 5' end is less
                    stable than its 3' end is loaded into RISC preferentially
                    (Khvorova et al. 2003; Schwarz et al. 2003).
3. Design rules   - the Reynolds et al. (2004) checklist, which is what a bench
                    scientist would apply by hand.

Sequences are guide (antisense) strands written 5'->3' in RNA letters (A, C, G, U).
"""

import numpy as np

BASES = "ACGU"
GUIDE_LENGTH = 19

# RNA/RNA nearest-neighbour duplex initiation + stacking free energies at 37 C
# (kcal/mol), Xia et al. (1998) Biochemistry 37:14719. Key = 5'->3' dinucleotide
# on one strand, paired with its complement.
NN_FREE_ENERGY = {
    "AA": -0.93, "AU": -1.10, "UA": -1.33, "CU": -2.08, "CA": -2.11,
    "GU": -2.24, "GA": -2.35, "CG": -2.36, "GG": -3.26, "GC": -3.42,
    "UU": -0.93, "AG": -2.08, "AC": -2.24, "UG": -2.11, "UC": -2.35,
    "CC": -3.26, "GAU": 0.0,  # placeholder key never used; kept out of lookups
}
HELIX_INIT = 4.09  # duplex initiation penalty
AU_END_PENALTY = 0.45  # per terminal AU or GU pair

# Reynolds et al. (2004) rules, expressed on the SENSE strand. We work with the
# guide, so each rule is applied to the complementary position of the guide.
REYNOLDS_RULES = [
    ("gc_30_52", "GC content between 30% and 52%"),
    ("a_at_sense_19", "A at sense position 19 (U at guide position 1)"),
    ("a_at_sense_3", "A at sense position 3"),
    ("u_at_sense_10", "U at sense position 10"),
    ("no_gc_at_sense_19", "No G/C at sense position 19"),
    ("no_g_at_sense_13", "No G at sense position 13"),
    ("au_rich_guide_5p", "AU-rich guide 5' end (first 5 bases)"),
]
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


class SequenceError(ValueError):
    """Raised when a sequence is not a usable siRNA guide strand."""


def clean_sequence(seq, length=GUIDE_LENGTH, name="sequence"):
    """Uppercase, convert DNA T to RNA U, strip whitespace, and validate."""
    if not isinstance(seq, str):
        raise SequenceError(f"{name} must be text")
    cleaned = "".join(seq.split()).upper().replace("T", "U")
    if not cleaned:
        raise SequenceError(f"{name} is empty")
    bad = sorted(set(cleaned) - set(BASES))
    if bad:
        raise SequenceError(f"{name} contains invalid letters: {', '.join(bad)}")
    if length is not None and len(cleaned) != length:
        raise SequenceError(f"{name} must be {length} nucleotides long, got {len(cleaned)}")
    return cleaned


def reverse_complement(seq):
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def duplex_free_energy(seq):
    """
    Nearest-neighbour free energy (kcal/mol) of `seq` paired with its complement.
    More negative means a more stable duplex.
    """
    if len(seq) < 2:
        return 0.0
    energy = HELIX_INIT + sum(NN_FREE_ENERGY[seq[i:i + 2]] for i in range(len(seq) - 1))
    for end in (seq[0], seq[-1]):
        if end in "AU":
            energy += AU_END_PENALTY
    return energy


def end_asymmetry(guide, window=5):
    """
    Difference in stability between the guide's two ends.

    Positive means the 5' end is less stable than the 3' end, which favours the
    guide strand being loaded into RISC. This is the single most informative
    thermodynamic feature in the published design rules.
    """
    return duplex_free_energy(guide[:window]) - duplex_free_energy(guide[-window:])


def reynolds_rules(guide):
    """Check the Reynolds design rules. Returns {rule_name: bool}."""
    sense = reverse_complement(guide)  # sense strand, 5'->3'
    gc = gc_content(guide)
    return {
        "gc_30_52": 0.30 <= gc <= 0.52,
        "a_at_sense_19": sense[18] == "A",
        "a_at_sense_3": sense[2] == "A",
        "u_at_sense_10": sense[9] == "U",
        "no_gc_at_sense_19": sense[18] not in "GC",
        "no_g_at_sense_13": sense[12] != "G",
        "au_rich_guide_5p": sum(b in "AU" for b in guide[:5]) >= 3,
    }


def reynolds_score(guide):
    """How many of the Reynolds rules this guide satisfies (0-7)."""
    return int(sum(reynolds_rules(guide).values()))


def feature_names():
    names = [f"pos{i + 1}_{b}" for i in range(GUIDE_LENGTH) for b in BASES]
    names += [f"freq_{b}" for b in BASES]
    names += [f"di_{a}{b}" for a in BASES for b in BASES]
    names += ["gc_content", "dg_total", "dg_5p_end", "dg_3p_end", "end_asymmetry",
              "reynolds_score"]
    return names


def featurize(guide):
    """Return the feature vector for one cleaned guide sequence."""
    values = []
    for i in range(GUIDE_LENGTH):
        values.extend(1.0 if guide[i] == b else 0.0 for b in BASES)
    values.extend(guide.count(b) / GUIDE_LENGTH for b in BASES)
    dinucleotides = [guide[i:i + 2] for i in range(GUIDE_LENGTH - 1)]
    values.extend(dinucleotides.count(a + b) / (GUIDE_LENGTH - 1) for a in BASES for b in BASES)
    values.extend([
        gc_content(guide),
        duplex_free_energy(guide),
        duplex_free_energy(guide[:5]),
        duplex_free_energy(guide[-5:]),
        end_asymmetry(guide),
        float(reynolds_score(guide)),
    ])
    return np.array(values, dtype=float)


def featurize_many(guides):
    """Feature matrix for a list of cleaned guide sequences."""
    return np.vstack([featurize(g) for g in guides])


def scan_transcript(mrna, length=GUIDE_LENGTH):
    """
    Walk a target mRNA and return every possible siRNA.

    For each window the guide is the reverse complement of the mRNA window, which
    is what gets loaded into RISC and binds the transcript.
    """
    mrna = clean_sequence(mrna, length=None, name="mRNA")
    if len(mrna) < length:
        raise SequenceError(f"mRNA must be at least {length} nucleotides long")
    return [
        {"position": i + 1, "target_site": mrna[i:i + length],
         "guide": reverse_complement(mrna[i:i + length])}
        for i in range(len(mrna) - length + 1)
    ]
