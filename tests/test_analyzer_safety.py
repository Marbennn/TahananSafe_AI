from inference.analyzer import IncidentAnalyzer


def _analyzer() -> IncidentAnalyzer:
    # Keep one analyzer instance per test call for deterministic behavior.
    return IncidentAnalyzer()


def test_physical_abuse_is_not_dropped() -> None:
    result = _analyzer().analyze(
        "My husband dragged me by the hair and punched me repeatedly in our house."
    )
    assert result["incident_type"] == "Physical Abuse"
    assert result["submission_decision"] == "ALLOW"
    assert result["allow_submission"] is True


def test_sexual_abuse_is_not_dropped_without_explicit_relation_terms() -> None:
    result = _analyzer().analyze(
        "Pinilit niya akong makipagtalik kahit tumatanggi ako."
    )
    assert result["incident_type"] == "Sexual Abuse"
    assert result["submission_decision"] == "ALLOW"
    assert result["allow_submission"] is True


def test_psychological_abuse_is_not_dropped() -> None:
    result = _analyzer().analyze(
        "Palagi niya akong pinagbabantaan at minumura sa bahay."
    )
    assert result["incident_type"] == "Psychological Abuse"
    assert result["submission_decision"] == "ALLOW"
    assert result["allow_submission"] is True


def test_neglect_abuse_is_not_dropped() -> None:
    result = _analyzer().analyze(
        "Walang nagbabantay sa bata at hindi siya pinapakain sa bahay."
    )
    assert result["incident_type"] == "Neglect / Acts of Omission"
    assert result["submission_decision"] == "ALLOW"
    assert result["allow_submission"] is True


def test_nonviolent_ambiguous_wording_is_blocked() -> None:
    result = _analyzer().analyze(
        "Sinaksak ko yung charger sa saksakan kasi lowbat ang phone ko."
    )
    assert result["incident_type"] == "None / Invalid"
    assert result["submission_decision"] == "BLOCKED"
    assert result["allow_submission"] is False


def test_surreal_non_abuse_report_is_blocked() -> None:
    result = _analyzer().analyze(
        "The electric fan threatened the television last night."
    )
    assert result["incident_type"] == "None / Invalid"
    assert result["submission_decision"] == "BLOCKED"
    assert result["allow_submission"] is False
