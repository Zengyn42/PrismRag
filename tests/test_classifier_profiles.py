"""Tests for classifier_profiles config field."""
from __future__ import annotations

from prism_rag.config import ClassifierProfile, PrismRagSettings, get_classifier_profile


def test_default_profile_present():
    s = PrismRagSettings()
    assert "default" in s.classifier_profiles
    assert s.classifier_profiles["default"].tier1_min_conf > 0


def test_lookup_known_model():
    s = PrismRagSettings()
    profile = get_classifier_profile(s, "bge-m3")
    assert profile.tier1_min_conf == 0.75
    assert profile.tier1_top_k == 1
    assert profile.tier1_min_consecutive == 2
    assert profile.tier2_min_conf == 0.70
    assert profile.tier2_margin == 0.25
    assert profile.tier2_hard_cap == 5


def test_lookup_unknown_model_falls_back_to_default():
    s = PrismRagSettings()
    profile = get_classifier_profile(s, "unknown-model-xyz")
    default = s.classifier_profiles["default"]
    assert profile == default
