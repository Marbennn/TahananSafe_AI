"""
Risk Scoring Module
Calculates risk percentage and determines risk level based on incident description
"""

import re
from typing import Dict, List, Tuple


class RiskScorer:
    """Calculates risk scores for incident reports"""
    
    # High-risk keywords (weighted by severity)
    CRITICAL_KEYWORDS = {
        'kill': 30, 'murder': 30, 'death': 25, 'dying': 25,
        'weapon': 20, 'gun': 25, 'knife': 20, 'blade': 20,
        'blood': 20, 'bleeding': 20, 'unconscious': 25,
        'hospital': 15, 'emergency': 15, 'ambulance': 15,
        'threaten': 15, 'threat': 15, 'kill you': 30,
        'stab': 25, 'stabbed': 28, 'stabbing': 28,
    }
    
    HIGH_RISK_KEYWORDS = {
        'violence': 15, 'violent': 15, 'attack': 15, 'assault': 15,
        'hurt': 12, 'injured': 12, 'injury': 12, 'wound': 12,
        'beat': 15, 'beating': 15, 'hit': 10, 'punch': 12,
        'strangle': 20, 'choke': 20, 'suffocate': 20,
        'rape': 25, 'sexual': 15, 'molest': 20,
        'children': 15, 'child': 15, 'kid': 15, 'minor': 15,
        'fear': 10, 'afraid': 10, 'scared': 10, 'terrified': 12
    }
    
    MEDIUM_RISK_KEYWORDS = {
        'push': 8, 'shove': 8, 'grab': 6, 'pull': 6,
        'yell': 5, 'scream': 5, 'shout': 5, 'angry': 6,
        'threat': 8, 'warning': 5, 'danger': 8,
        'control': 5, 'restrict': 5, 'isolate': 6,
        'money': 4, 'financial': 4, 'steal': 6, 'theft': 6
    }
    
    # Tagalog keywords
    TAGALOG_CRITICAL = {
        # Direct or explicit threats to kill / death / severe harm
        'patayin': 30, 'papatayin': 30,
        'patay': 25, 'namatay': 25,
        'sugat': 20, 'dugo': 20, 'duguan': 20, 'wala sa malay': 25,
        'baril': 25, 'kutsilyo': 20, 'kutilyo': 20, 'armas': 20,
        # Stabbing / cutting (implies weapon + serious violence)
        'sinaksak': 28, 'saksak': 28, 'tinaga': 22,
    }
    
    TAGALOG_HIGH = {
        'karahasan': 15, 'saktan': 12, 'bugbog': 15,
        'suntok': 12, 'sinuntok': 15, 'sinusuntok': 18,
        'bata': 15, 'anak': 12,
        'takot': 10, 'natatakot': 12, 'tinatakot': 14
    }
    
    def __init__(self):
        """Initialize risk scorer"""
        pass
    
    def calculate_risk_percentage(self, incident_description: str) -> float:
        """
        Calculate risk percentage (0-100) based on incident description
        
        Args:
            incident_description: Text description of the incident
            
        Returns:
            Risk percentage as float
        """
        if not incident_description:
            return 0.0
        
        text_lower = incident_description.lower()
        risk_score = 0.0
        
        # Check critical keywords
        for keyword, weight in self.CRITICAL_KEYWORDS.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            risk_score += count * weight
        
        # Check high-risk keywords
        for keyword, weight in self.HIGH_RISK_KEYWORDS.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            risk_score += count * weight
        
        # Check medium-risk keywords
        for keyword, weight in self.MEDIUM_RISK_KEYWORDS.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            risk_score += count * weight
        
        # Check Tagalog keywords
        for keyword, weight in self.TAGALOG_CRITICAL.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            risk_score += count * weight
        
        for keyword, weight in self.TAGALOG_HIGH.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            risk_score += count * weight
        
        # Normalize to 0-100 range
        # Base risk should be 0 so negative/irrelevant reports stay near 0.
        base_risk = 0.0
        
        # Cap maximum score contribution
        max_contribution = 90.0
        risk_score = min(risk_score, max_contribution)
        
        total_risk = base_risk + risk_score
        total_risk = min(total_risk, 100.0)
        total_risk = max(total_risk, 0.0)
        
        return round(total_risk, 2)
    
    def determine_risk_level(self, risk_percentage: float) -> str:
        """
        Determine risk level based on percentage
        
        Args:
            risk_percentage: Calculated risk percentage
            
        Returns:
            Risk level: 'Low', 'Medium', 'High', or 'Critical'
        """
        if risk_percentage >= 80:
            return 'Critical'
        elif risk_percentage >= 60:
            return 'High'
        elif risk_percentage >= 40:
            return 'Medium'
        else:
            return 'Low'
    
    def determine_priority_level(self, risk_percentage: float, risk_level: str) -> str:
        """
        Determine priority level based on risk
        
        Args:
            risk_percentage: Calculated risk percentage
            risk_level: Determined risk level
            
        Returns:
            Priority level: 'First Priority (P1)', 'Second Priority (P2)', or 'Third Priority (P3)'
        """
        if risk_level == 'Critical' or risk_percentage >= 80:
            return 'First Priority (P1)'
        elif risk_level == 'High' or risk_percentage >= 60:
            return 'Second Priority (P2)'
        else:
            return 'Third Priority (P3)'

    def adjust_with_context(
        self,
        base_risk: float,
        incident_type: str,
        children_involved: bool,
        weapon_mentioned: bool
    ) -> float:
        """
        Adjust an initial risk score using contextual factors derived
        from previous analyses (incident type, children, weapons).

        This acts like a simple formula the system can reuse across
        incidents, so similar patterns get similar risk adjustments.
        """
        risk = float(base_risk)

        # Baseline by incident type (heavier floor for some categories)
        if incident_type == "Sexual Abuse":
            risk = max(risk, 70.0)
        elif incident_type in {"Physical Abuse", "Elder Abuse"}:
            risk = max(risk, 50.0)
        elif incident_type in {"Psychological Abuse", "Economic Abuse"}:
            risk = max(risk, 35.0)

        # Children increase urgency
        if children_involved:
            risk += 10.0

        # Weapons significantly increase risk
        if weapon_mentioned:
            risk += 15.0

        # Clamp to 0-100
        risk = max(0.0, min(risk, 100.0))
        return round(risk, 2)

