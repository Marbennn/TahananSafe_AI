"""
Validation utilities for incident reports
"""

from typing import Dict, List, Any, Optional


class IncidentValidator:
    """Validates incident report data"""
    
    # Core abuse categories (can be combined for multi-label).
    ABUSE_CORE_TYPES = [
        "Physical Abuse",
        "Sexual Abuse",
        "Psychological Abuse",
        "Economic Abuse",
        "Elder Abuse",
        "Neglect / Acts of Omission",
    ]

    # Non-abuse but valid barangay/community report types.
    COMMUNITY_REPORT_TYPES = [
        "Theft / Robbery",
        "Physical Altercation / Assault",
        "Community or Neighbor Disputes",
        "Public Disturbance",
        "Missing Person",
        "Property Damage / Vandalism",
        "Fraud / Scams",
        "Suspicious Activity",
        "Out-of-Scope Reports",
    ]

    # Allowed model/report output types for abuse-focused workflow.
    ABUSE_TYPES = ABUSE_CORE_TYPES + [
        # Negative / non-abuse bucket (from Negative_Dataset)
        "None / Invalid",
        "None / False Report",
        # Fallback bucket when the system is unsure
        "Unknown",
    ]
    
    LANGUAGES = [
        "English",
        "Tagalog",
        "Ilocano",
        "Pangasinan",
        "Mixed Language"
    ]
    
    RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
    
    PRIORITY_LEVELS = [
        "First Priority (P1)",
        "Second Priority (P2)",
        "Third Priority (P3)"
    ]
    
    BARANGAY_CATEGORIES = [
        "Domestic Violence / Family Violence",
        "Child Abuse / Child Neglect",
        "Harassment / Threats",
        "Theft / Robbery",
        "Physical Altercation / Assault",
        "Community or Neighbor Disputes",
        "Public Disturbance",
        "Missing Person",
        "Property Damage / Vandalism",
        "Fraud / Scams",
        "Suspicious Activity",
        "Out-of-Scope Reports",
    ]
    
    @staticmethod
    def validate_incident_description(description: str) -> tuple[bool, Optional[str]]:
        """Validate incident description"""
        if not description or not description.strip():
            return False, "Incident description is required"
        
        if len(description) < 10:
            return False, "Incident description must be at least 10 characters"
        
        if len(description) > 5000:
            return False, "Incident description must be less than 5000 characters"
        
        return True, None
    
    @staticmethod
    def validate_incident_type(incident_type: str) -> tuple[bool, Optional[str]]:
        """Validate incident type"""
        if incident_type not in IncidentValidator.ABUSE_TYPES:
            return False, f"Invalid incident type. Must be one of: {', '.join(IncidentValidator.ABUSE_TYPES)}"
        return True, None

    @staticmethod
    def validate_incident_types(incident_types: List[str]) -> tuple[bool, Optional[str]]:
        """
        Validate a list of incident types (multi-label).
        Rules:
        - Must be a non-empty list of allowed types.
        - Multi-label is only for abuse-core combinations.
        - Community report types are single-label only.
        - Do not mix 'None / Invalid' or 'Unknown' with other labels.
        """
        if not isinstance(incident_types, list) or not incident_types:
            return False, "incident_types must be a non-empty list"

        for t in incident_types:
            if t not in IncidentValidator.ABUSE_TYPES:
                return False, f"Invalid incident type in incident_types: {t}"

        if len(incident_types) > 1:
            invalid_mix = {"None / Invalid", "None / False Report", "Unknown"}
            if any(t in invalid_mix for t in incident_types):
                return False, "incident_types with multiple entries cannot include None/Invalid/Unknown buckets"
            if any(t in IncidentValidator.COMMUNITY_REPORT_TYPES for t in incident_types):
                return False, "incident_types cannot mix community report types with other labels"

        return True, None
    
    @staticmethod
    def validate_language(language: str) -> tuple[bool, Optional[str]]:
        """Validate language"""
        if language not in IncidentValidator.LANGUAGES:
            return False, f"Invalid language. Must be one of: {', '.join(IncidentValidator.LANGUAGES)}"
        return True, None
    
    @staticmethod
    def validate_risk_percentage(risk_percentage: float) -> tuple[bool, Optional[str]]:
        """Validate risk percentage"""
        if not isinstance(risk_percentage, (int, float)):
            return False, "Risk percentage must be a number"
        
        if risk_percentage < 0 or risk_percentage > 100:
            return False, "Risk percentage must be between 0 and 100"
        
        return True, None

    @staticmethod
    def validate_barangay_category(category: str) -> tuple[bool, Optional[str]]:
        """Validate barangay blotter category"""
        if category not in IncidentValidator.BARANGAY_CATEGORIES:
            return (
                False,
                "Invalid barangay_category. Must be one of: "
                + ", ".join(IncidentValidator.BARANGAY_CATEGORIES),
            )
        return True, None
    
    @staticmethod
    def validate_analysis_output(analysis: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate complete analysis output"""
        required_fields = [
            'incident_type',
            'incident_types',
            'language',
            'risk_level',
            'risk_percentage',
            'priority_level',
            'children_involved',
            'weapon_mentioned',
            'confidence_score'
        ]
        
        for field in required_fields:
            if field not in analysis:
                return False, f"Missing required field: {field}"
        
        # Validate individual fields
        # Primary incident_type
        valid, error = IncidentValidator.validate_incident_type(analysis['incident_type'])
        if not valid:
            return False, error

        # Multi-label incident_types
        valid, error = IncidentValidator.validate_incident_types(analysis['incident_types'])
        if not valid:
            return False, error
        
        valid, error = IncidentValidator.validate_language(analysis['language'])
        if not valid:
            return False, error
        
        if analysis['risk_level'] not in IncidentValidator.RISK_LEVELS:
            return False, f"Invalid risk level: {analysis['risk_level']}"
        
        valid, error = IncidentValidator.validate_risk_percentage(analysis['risk_percentage'])
        if not valid:
            return False, error
        
        if analysis['priority_level'] not in IncidentValidator.PRIORITY_LEVELS:
            return False, f"Invalid priority level: {analysis['priority_level']}"
        
        if not isinstance(analysis['children_involved'], bool):
            return False, "children_involved must be a boolean"
        
        if not isinstance(analysis['weapon_mentioned'], bool):
            return False, "weapon_mentioned must be a boolean"
        
        if not isinstance(analysis['confidence_score'], (int, float)):
            return False, "confidence_score must be a number"
        
        if analysis['confidence_score'] < 0 or analysis['confidence_score'] > 100:
            return False, "confidence_score must be between 0 and 100"

        if "incident_tip" in analysis and (
            not isinstance(analysis["incident_tip"], str) or not analysis["incident_tip"].strip()
        ):
            return False, "incident_tip must be a non-empty string"

        if "allow_submission" in analysis and not isinstance(analysis["allow_submission"], bool):
            return False, "allow_submission must be a boolean"

        if "submission_decision" in analysis:
            if not isinstance(analysis["submission_decision"], str) or not analysis["submission_decision"].strip():
                return False, "submission_decision must be a non-empty string"
            if analysis["submission_decision"] not in {"ALLOW", "BLOCKED"}:
                return False, "submission_decision must be ALLOW or BLOCKED"

        if "validation_reason" in analysis and (
            not isinstance(analysis["validation_reason"], str) or not analysis["validation_reason"].strip()
        ):
            return False, "validation_reason must be a non-empty string"

        if "barangay_category" in analysis:
            if not isinstance(analysis["barangay_category"], str) or not analysis["barangay_category"].strip():
                return False, "barangay_category must be a non-empty string"
            valid, error = IncidentValidator.validate_barangay_category(analysis["barangay_category"])
            if not valid:
                return False, error

        if "barangay_category_confidence" in analysis:
            val = analysis["barangay_category_confidence"]
            if not isinstance(val, (int, float)):
                return False, "barangay_category_confidence must be a number"
            if val < 0 or val > 100:
                return False, "barangay_category_confidence must be between 0 and 100"

        if "abuse_related" in analysis and not isinstance(analysis["abuse_related"], bool):
            return False, "abuse_related must be a boolean"
        
        return True, None
