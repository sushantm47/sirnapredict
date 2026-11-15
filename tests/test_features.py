"""Tests for sequence featurization and the biology it encodes."""

import pytest

from src import features as F

GUIDE = "UUGAUAUGUUGGAUGAUGG"


# --------------------------------------------------------------- cleaning

def test_clean_accepts_dna_and_whitespace():
    assert F.clean_sequence(" uugauaugttggaugaugg ") == GUIDE


@pytest.mark.parametrize("seq, message", [
    ("UUGAUAUGUUGGAUGAUG", "19 nucleotides"),      # too short
    ("UUGAUAUGUUGGAUGAUGGA", "19 nucleotides"),    # too long
    ("UUGAUAUGUUGGAUGAUNN", "invalid letters"),
    ("", "empty"),
    (12345, "must be text"),
])
def test_clean_rejects_bad_input(seq, message):
    with pytest.raises(F.SequenceError, match=message):
        F.clean_sequence(seq)


def test_reverse_complement_is_its_own_inverse():
    assert F.reverse_complement(F.reverse_complement(GUIDE)) == GUIDE
    assert F.reverse_complement("AAAAACCCCCGGGGGUUUU") == "AAAACCCCCGGGGGUUUUU"


# --------------------------------------------------------------- thermodynamics

def test_gc_rich_duplex_is_more_stable():
    gc_rich = F.duplex_free_energy("GCGCGCGCGCGCGCGCGCG")
    au_rich = F.duplex_free_energy("AUAUAUAUAUAUAUAUAUA")
    assert gc_rich < au_rich < 0


def test_end_asymmetry_sign_and_antisymmetry():
    # AU-rich 5' end, GC-rich 3' end -> positive, which favours RISC loading
    guide = "AUAUAUAUAUAUAUGCGCG"
    assert F.end_asymmetry(guide) > 0
    assert F.end_asymmetry(F.reverse_complement(guide)) == pytest.approx(-F.end_asymmetry(guide))


def test_asymmetry_is_the_difference_between_the_two_ends():
    guide = "AUAUAUAUAUAUAUGCGCG"
    expected = F.duplex_free_energy(guide[:5]) - F.duplex_free_energy(guide[-5:])
    assert F.end_asymmetry(guide) == pytest.approx(expected)


def test_equally_stable_ends_give_zero_asymmetry():
    assert F.end_asymmetry("ACGUAUUUUUUUUUACGUA") == pytest.approx(0.0)


# --------------------------------------------------------------- design rules

def test_rules_are_all_checked():
    rules = F.reynolds_rules(GUIDE)
    assert set(rules) == {name for name, _ in F.REYNOLDS_RULES}
    assert F.reynolds_score(GUIDE) == sum(rules.values())


def test_gc_rule_tracks_gc_content():
    assert F.reynolds_rules("AUAUAUAUAUAUAUAUAUA")["gc_30_52"] is False   # 0% GC
    assert F.reynolds_rules("GCGCGCGCGCGCGCGCGCG")["gc_30_52"] is False   # 100% GC
    assert F.reynolds_rules(GUIDE)["gc_30_52"] is True                    # 37% GC


# --------------------------------------------------------------- feature vector

def test_feature_vector_matches_names_and_is_finite():
    import numpy as np
    vector = F.featurize(GUIDE)
    assert len(vector) == len(F.feature_names()) == 102
    assert np.isfinite(vector).all()


def test_position_one_hot_encodes_the_sequence():
    vector = F.featurize(GUIDE)
    for i, base in enumerate(GUIDE):
        block = vector[i * 4:(i + 1) * 4]
        assert block.sum() == 1
        assert block[F.BASES.index(base)] == 1


def test_featurize_many_stacks_rows():
    assert F.featurize_many([GUIDE, F.reverse_complement(GUIDE)]).shape == (2, 102)


# --------------------------------------------------------------- transcript scan

def test_scan_returns_every_window_with_matching_guides():
    mrna = "AUGGCCUUGAUAUGUUGGAUGAUGGAACCUUGG"
    windows = F.scan_transcript(mrna)
    assert len(windows) == len(mrna) - 18
    assert windows[0]["position"] == 1
    for w in windows:
        assert w["guide"] == F.reverse_complement(w["target_site"])
        assert mrna[w["position"] - 1:w["position"] + 18] == w["target_site"]


def test_scan_rejects_short_transcript():
    with pytest.raises(F.SequenceError, match="at least 19"):
        F.scan_transcript("AUGGCC")
