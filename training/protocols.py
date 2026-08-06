"""Attack protocols used in the TaFD paper."""

from __future__ import annotations

from collections.abc import Mapping


# The source order is kept compatible with the released K=2 checkpoints.
# During training, the first two sources are generated with 10-step PGD/PGD-L2;
# their historical source identifiers remain unchanged so existing assignment
# indices and checkpoint state remain compatible.
CANONICAL_ATTACK_SOURCE_ORDER = (
    "APGD_Linf",
    "APGD_L2",
    "ACE",
    "HSVAdv",
    "ReColorAdv",
    "ALA",
    "RetouchUAA",
)

# Evaluation follows the display order used in the paper. Source identifiers
# remain governed by CANONICAL_ATTACK_SOURCE_ORDER for checkpoint compatibility.
CANONICAL_ATTACK_TEST_ORDER = (
    "Clean",
    "APGD_Linf",
    "APGD_L2",
    "ACE",
    "ALA",
    "HSVAdv",
    "ReColorAdv",
    "RetouchUAA",
)

BROADER_ATTACK_SOURCE_ORDER = (
    "APGD_Linf",
    "APGD_L2",
    "ACE",
    "GPGD",
    "StAdv",
)

BROADER_ATTACK_TEST_ORDER = (
    "Clean",
    "APGD_Linf",
    "APGD_L2",
    "ACE",
    "StAdv",
    "GPGD",
)

ATTACK_UNIONS = {
    "canonical": {
        "train_attacks": list(CANONICAL_ATTACK_SOURCE_ORDER),
        "test_attacks": list(CANONICAL_ATTACK_TEST_ORDER),
        "num_attack_sources": len(CANONICAL_ATTACK_SOURCE_ORDER),
        "attack_names": list(CANONICAL_ATTACK_SOURCE_ORDER),
    },
    "broader": {
        "train_attacks": list(BROADER_ATTACK_SOURCE_ORDER),
        "test_attacks": list(BROADER_ATTACK_TEST_ORDER),
        "num_attack_sources": len(BROADER_ATTACK_SOURCE_ORDER),
        "attack_names": list(BROADER_ATTACK_SOURCE_ORDER),
    },
}

ALL_ATTACKS = list(
    dict.fromkeys((*CANONICAL_ATTACK_TEST_ORDER, *BROADER_ATTACK_TEST_ORDER))
)

DIAGNOSIS_ATTACKS = [name for name in ALL_ATTACKS if name != "Clean"]
DIAGNOSIS_SUPERVISED_ATTACKS = list(DIAGNOSIS_ATTACKS)

ATTACK_SHORT_NAMES = {
    "APGD_Linf": "AP_L",
    "APGD_L2": "AP_2",
    "ACE": "ACE",
    "HSVAdv": "HSV",
    "ReColorAdv": "ReC",
    "ALA": "ALA",
    "RetouchUAA": "RetUAA",
    "GPGD": "GPGD",
    "StAdv": "STA",
    "Clean": "Clean",
}

LEGACY_ATTACK_NAME_ALIASES = {
    "Hue": "HSVAdv",
    "Light": "ALA",
    "UAA": "RetouchUAA",
}


def paper_attack_name(name: str) -> str:
    """Return the paper-aligned name for a historical or current attack key."""
    return LEGACY_ATTACK_NAME_ALIASES.get(name, name)


def migrate_metric_history(history):
    """Translate legacy attack keys when an earlier checkpoint is resumed."""
    if not isinstance(history, Mapping):
        return history

    migrated = {}
    for name, values in history.items():
        migrated_name = paper_attack_name(name)
        if migrated_name not in migrated or migrated_name == name:
            migrated[migrated_name] = values
    return migrated
