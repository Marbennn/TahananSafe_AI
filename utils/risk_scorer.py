"""
Risk Scoring Module
Calculates risk percentage and determines risk level based on incident description
"""

import re
from typing import Any


class RiskScorer:
    """Calculates risk scores for incident reports"""
    
    # High-risk keywords (weighted by severity)
    CRITICAL_KEYWORDS = {
        'kill': 30, 'murder': 30, 'death': 25, 'dying': 25,
        'weapon': 20, 'gun': 25, 'knife': 20, 'blade': 20,
        'blood': 20, 'bleeding': 20, 'unconscious': 25,
        'broken bone': 28, 'broken bones': 30, 'fracture': 28, 'fractured': 28,
        'cannot breathe': 28, "can't breathe": 28, 'no breath': 28,
        'hospital': 15, 'emergency': 15, 'ambulance': 15,
        'threaten': 15, 'threat': 15, 'kill you': 30,
        'stab': 25, 'stabbed': 28, 'stabbing': 28,
        'shoot': 24, 'shot': 24, 'shooting': 24,
    }
    
    HIGH_RISK_KEYWORDS = {
        'violence': 15, 'violent': 15, 'attack': 15, 'assault': 15,
        'hurt': 12, 'injured': 12, 'injury': 12, 'wound': 12,
        'beat': 15, 'beating': 15, 'hit': 10, 'punch': 12,
        'drag': 14, 'dragged': 16, 'hair': 12,
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
        'nabalian': 28, 'nawalan ng hininga': 30, 'walang hininga': 30,
        'baril': 25, 'kutsilyo': 20, 'kutilyo': 20, 'armas': 20,
        # Stabbing / cutting (implies weapon + serious violence)
        'sinaksak': 28, 'saksak': 28, 'tinaga': 22,
        'binaril': 28, 'barilin': 28, 'pinagbabaril': 30,
    }
    
    TAGALOG_HIGH = {
        'karahasan': 15, 'saktan': 12, 'bugbog': 15,
        'suntok': 12, 'sinuntok': 15, 'sinusuntok': 18,
        'hila': 12, 'hinila': 14, 'kinaladkad': 16,
        'hampas': 14, 'hinampas': 16,
        'tulak': 12, 'tinulak': 16, 'itinulak': 16,
        'nahulog': 16, 'nalaglag': 16,
        'bata': 15, 'anak': 12,
        'takot': 10, 'natatakot': 12, 'tinatakot': 14
    }

    SAKSAK_TERMS = {"sinaksak", "saksak", "stab", "stabbed", "stabbing", "tinaga"}
    KILL_TERMS = {"kill", "killed", "killing", "patay", "pinatay", "patayin", "papatayin"}
    SHOOT_TERMS = {"shoot", "shot", "shooting", "binaril", "barilin", "pinagbabaril"}
    IMPACT_TERMS = {
        "hit", "hits", "hitting", "punch", "punched", "punching",
        "slap", "slapped", "slapping", "kick", "kicked", "kicking",
        "drag", "dragged", "dragging", "pull", "pulled", "pulling",
        "suntok", "sinuntok", "sinusuntok", "sampal", "sinampal",
        "hampas", "hinampas", "hinampasan", "sipa", "sinipa",
        "hila", "hinila", "kinaladkad", "kaladkad",
        "bugbog", "binugbog", "saktan", "sinaktan",
        "tulak", "tinulak", "itinulak", "push", "pushed",
    }

    CHILD_TERMS = {
        "baby", "infant", "newborn", "toddler", "child", "children",
        "kid", "kids", "minor", "minors",
        "bata", "mga bata", "anak", "mga anak",
        "sanggol", "apo", "binata", "dalagita",
        "ubing", "mga ubing",
        "ugaw", "mga ugaw",
    }

    CHILD_VIOLENCE_TERMS = (
        IMPACT_TERMS
        .union(SAKSAK_TERMS)
        .union(SHOOT_TERMS)
        .union({"choke", "strangle", "sinakal", "sakal", "kill", "patayin", "papatayin"})
    )

    NON_VIOLENT_SAKSAK_TARGETS = {
        "phone", "cellphone", "cp", "charger", "charge", "charging",
        "usb", "cable", "kable", "socket", "outlet", "extension", "saksakan",
        "adapter", "powerbank", "kuryente", "plug", "plugged", "laptop", "tv",
    }
    NON_VIOLENT_KILL_TARGETS = {
        "ilaw", "lights", "light", "lamp", "tv", "aircon", "fan",
        "wifi", "data", "internet", "bluetooth", "alarm", "app",
        "switch", "engine", "motor", "machine", "music", "video",
    }
    NON_VIOLENT_SHOOT_TARGETS = {
        "camera", "photo", "picture", "video", "film", "movie",
        "basketball", "ball", "hoop", "goal", "game", "content",
    }
    NON_VIOLENT_IMPACT_TARGETS = {
        "pader", "wall", "door", "window", "mesa", "table", "chair", "upuan",
        "punching bag", "boxing bag", "bag", "keyboard", "laptop", "phone",
        "cellphone", "ball", "bola", "pillow", "unan", "object",
    }
    NON_VIOLENT_SAKSAK_PHRASES = {
        "mag charge", "mag-charge", "nag charge", "nakacharge", "naka charge",
        "for charging", "to charge", "isaksak sa saksakan",
    }
    NON_VIOLENT_KILL_PHRASES = {
        "turn off", "turned off", "shut down", "shutdown",
        "pinatay ko ang ilaw", "patayin ang ilaw", "patay sindi",
    }
    NON_VIOLENT_SHOOT_PHRASES = {
        "photo shoot", "shoot ng video", "shoot ng content", "camera shoot",
    }
    NON_VIOLENT_IMPACT_PHRASES = {
        "boxing training", "nag-boxing", "punching bag", "sinampal ng hangin",
    }
    IMPROVISED_WEAPON_OBJECTS = {
        "tinidor", "fork", "plato", "plate", "baso", "glass",
        "bote", "botelya", "bottle", "screwdriver", "distornilyador",
        "kadena", "chain", "brick", "ladrilyo", "rock", "bato",
    }
    BODY_PART_TERMS = {
        "ulo", "head", "mukha", "face", "leeg", "neck",
        "likod", "back", "dibdib", "chest", "tiyan", "stomach",
    }
    HUMAN_CONTEXT_TERMS = {
        "ako", "siya", "sya", "kami", "kayo", "sila",
        "me", "my", "him", "her", "them", "victim", "biktima",
        "tatay", "nanay", "asawa", "anak", "kapatid", "lolo", "lola",
        "father", "mother", "husband", "wife", "partner",
    }
    VICTIM_CONTEXT_TERMS = {
        "ako", "siya", "sya",
        "me", "him", "her", "victim", "biktima",
        "tatay", "nanay", "asawa", "anak", "kapatid", "lolo", "lola",
        "father", "mother", "husband", "wife", "partner",
    }
    INJURY_CONTEXT_TERMS = {
        "dugo", "duguan", "blood", "bleeding", "injury", "injured",
        "wound", "sugat", "hospital", "ambulance", "nasaktan",
        "fracture", "fractured", "broken bone", "broken bones",
        "nabalian", "nawalan ng hininga", "walang hininga",
        "could not stand", "unable to stand", "cannot stand", "hindi makatayo",
    }
    
    def __init__(self):
        """Initialize risk scorer"""
        pass

    @staticmethod
    def _normalize_noisy_text(text: str) -> str:
        """Normalize noisy spelling variants (e.g., repeated letters)."""
        if not text:
            return ""
        normalized = text.lower()
        normalized = re.sub(r"([a-zA-Z])\1{2,}", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _keyword_count(self, text: str, keyword: str, normalized_text: str = "") -> int:
        """Count keyword hits, with fallback to normalized noisy text."""
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count = len(re.findall(pattern, text))
        if count == 0 and normalized_text and normalized_text != text:
            count = len(re.findall(pattern, normalized_text))
        return count

    def _contains_keyword(self, text: str, keyword: str) -> bool:
        if not text or not keyword:
            return False
        text_lower = text.lower()
        kw = keyword.lower().strip()
        pattern = rf'(?<!\w){re.escape(kw)}(?!\w)'
        if re.search(pattern, text_lower) is not None:
            return True
        text_norm = self._normalize_noisy_text(text_lower)
        if text_norm != text_lower and re.search(pattern, text_norm) is not None:
            return True
        if re.fullmatch(r"[a-z]{4,}", kw):
            inflected = rf'(?<!\w){re.escape(kw)}(?:s|es|ed|ing)?(?!\w)'
            if re.search(inflected, text_lower) is not None:
                return True
            if text_norm != text_lower and re.search(inflected, text_norm) is not None:
                return True
        return False

    def _has_proximity_match(
        self,
        text: str,
        terms_a: set[str] | list[str],
        terms_b: set[str] | list[str],
        max_gap_chars: int = 45,
    ) -> bool:
        if not text:
            return False

        variants = [text.lower()]
        text_norm = self._normalize_noisy_text(text)
        if text_norm and text_norm not in variants:
            variants.append(text_norm)

        for text_lower in variants:
            for a in terms_a:
                for b in terms_b:
                    pattern = (
                        rf'(?<!\w){re.escape(a.lower())}(?!\w).{{0,{max_gap_chars}}}(?<!\w){re.escape(b.lower())}(?!\w)|'
                        rf'(?<!\w){re.escape(b.lower())}(?!\w).{{0,{max_gap_chars}}}(?<!\w){re.escape(a.lower())}(?!\w)'
                    )
                    if re.search(pattern, text_lower):
                        return True
        return False

    def _is_nonviolent_action_context(
        self,
        text: str,
        action_terms: set[str] | list[str],
        nonviolent_targets: set[str] | list[str],
        nonviolent_phrases: set[str] | list[str],
        max_gap_chars: int = 45,
    ) -> bool:
        """Detect non-violent context for ambiguous action words."""
        if not text:
            return False
        text_lower = text.lower()
        has_action = any(self._contains_keyword(text_lower, term) for term in action_terms)
        if not has_action:
            return False

        near_nonviolent_target = self._has_proximity_match(
            text_lower,
            action_terms,
            nonviolent_targets,
            max_gap_chars=max_gap_chars,
        )
        has_nonviolent_phrase = any(phrase in text_lower for phrase in nonviolent_phrases)

        has_harm_signal = self._has_proximity_match(
            text_lower,
            action_terms,
            self.VICTIM_CONTEXT_TERMS.union(self.INJURY_CONTEXT_TERMS),
            max_gap_chars=max_gap_chars,
        )

        return (near_nonviolent_target or has_nonviolent_phrase) and not has_harm_signal

    def _is_nonviolent_saksak_context(self, text: str) -> bool:
        return self._is_nonviolent_action_context(
            text,
            action_terms=self.SAKSAK_TERMS,
            nonviolent_targets=self.NON_VIOLENT_SAKSAK_TARGETS,
            nonviolent_phrases=self.NON_VIOLENT_SAKSAK_PHRASES,
            max_gap_chars=45,
        )

    def _is_nonviolent_kill_context(self, text: str) -> bool:
        return self._is_nonviolent_action_context(
            text,
            action_terms=self.KILL_TERMS,
            nonviolent_targets=self.NON_VIOLENT_KILL_TARGETS,
            nonviolent_phrases=self.NON_VIOLENT_KILL_PHRASES,
            max_gap_chars=40,
        )

    def _is_nonviolent_shoot_context(self, text: str) -> bool:
        return self._is_nonviolent_action_context(
            text,
            action_terms=self.SHOOT_TERMS,
            nonviolent_targets=self.NON_VIOLENT_SHOOT_TARGETS,
            nonviolent_phrases=self.NON_VIOLENT_SHOOT_PHRASES,
            max_gap_chars=45,
        )

    def _is_nonviolent_impact_context(self, text: str) -> bool:
        return self._is_nonviolent_action_context(
            text,
            action_terms=self.IMPACT_TERMS,
            nonviolent_targets=self.NON_VIOLENT_IMPACT_TARGETS,
            nonviolent_phrases=self.NON_VIOLENT_IMPACT_PHRASES,
            max_gap_chars=35,
        )

    def _child_victim_violence_boost(self, text: str) -> float:
        """
        Strong risk boost when violent actions are directed at a child/baby.
        Example:
        - may sinuntok na baby
        - sinipa ang bata
        - hinampas ang anak
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        boost = 0.0

        has_child = any(self._contains_keyword(text_lower, term) for term in self.CHILD_TERMS)
        if not has_child:
            return 0.0

        nonviolent_saksak = self._is_nonviolent_saksak_context(text_lower)
        nonviolent_shoot = self._is_nonviolent_shoot_context(text_lower)
        nonviolent_impact = self._is_nonviolent_impact_context(text_lower)
        nonviolent_kill = self._is_nonviolent_kill_context(text_lower)

        violent_near_child = self._has_proximity_match(
            text_lower,
            self.CHILD_VIOLENCE_TERMS,
            self.CHILD_TERMS,
            max_gap_chars=45,
        )

        explicit_child_attack_patterns = [
            r"(sinuntok|suntok|binugbog|bugbog|sinipa|sipa|sinampal|sampal|hinampas|hampas|sinaktan|saktan).{0,45}(baby|bata|anak|sanggol|child|kid|minor|infant|toddler)",
            r"(baby|bata|anak|sanggol|child|kid|minor|infant|toddler).{0,45}(sinuntok|suntok|binugbog|bugbog|sinipa|sipa|sinampal|sampal|hinampas|hampas|sinaktan|saktan)",
            r"(sinaksak|saksak|tinaga|stab|stabbed|binaril|barilin|shoot|shot).{0,45}(baby|bata|anak|sanggol|child|kid|minor|infant|toddler)",
            r"(baby|bata|anak|sanggol|child|kid|minor|infant|toddler).{0,45}(sinaksak|saksak|tinaga|stab|stabbed|binaril|barilin|shoot|shot)",
        ]
        has_explicit_pattern = any(re.search(pattern, text_lower) for pattern in explicit_child_attack_patterns)

        if violent_near_child or has_explicit_pattern:
            boost += 28.0

        # Stronger for very young children.
        infant_terms = {"baby", "sanggol", "infant", "newborn", "toddler"}
        if any(self._contains_keyword(text_lower, term) for term in infant_terms) and (violent_near_child or has_explicit_pattern):
            boost += 10.0

        # Extra severity for extreme violence against a child.
        severe_terms = self.SAKSAK_TERMS.union(self.SHOOT_TERMS).union({"choke", "strangle", "sinakal", "patayin", "papatayin"})
        severe_near_child = self._has_proximity_match(
            text_lower,
            severe_terms,
            self.CHILD_TERMS,
            max_gap_chars=45,
        )
        if severe_near_child:
            boost += 12.0

        # Remove accidental boosts from clearly non-violent contexts.
        if nonviolent_impact and not severe_near_child:
            boost = max(0.0, boost - 18.0)
        if nonviolent_saksak:
            boost = max(0.0, boost - 20.0)
        if nonviolent_shoot:
            boost = max(0.0, boost - 20.0)
        if nonviolent_kill:
            boost = max(0.0, boost - 15.0)

        return boost

    def _severe_risk_boost(self, text: str) -> float:
        """Add extra risk for high-danger physical contexts not captured by simple keywords."""
        if not text:
            return 0.0
        text_lower = text.lower()
        boost = 0.0

        # Push/fall from height context (very high danger).
        if re.search(
            r"(tinulak|itinulak|pushed|push).{0,60}(pababa|nahulog|nalaglag|palapag|hagdan|floor|second floor|ikalawang palapag)",
            text_lower,
        ):
            boost += 45.0

        # Strike to critical body area (e.g., head).
        if self._has_proximity_match(
            text_lower,
            self.IMPACT_TERMS.union({"tinulak", "itinulak"}),
            self.BODY_PART_TERMS,
            max_gap_chars=35,
        ):
            boost += 18.0

        # Improvised object used in violent/threat context.
        improvised_context = self._has_proximity_match(
            text_lower,
            self.IMPACT_TERMS.union(self.SAKSAK_TERMS).union(self.SHOOT_TERMS).union({"banta", "pinagbantaan", "papatayin"}),
            self.IMPROVISED_WEAPON_OBJECTS,
            max_gap_chars=50,
        )
        if improvised_context:
            boost += 15.0

        # Severe injury outcomes (fracture, inability to stand/breathe).
        if re.search(
            r"(broken bones?|fracture|fractured|nabalian|nawalan ng hininga|walang hininga|could not stand|unable to stand|cannot stand|hindi makatayo|can't breathe|cannot breathe)",
            text_lower,
        ):
            boost += 35.0

        # Dragged by hair / dragged on floor indicates severe physical violence.
        if re.search(
            r"(drag|dragged|dragging|hinila|kinaladkad).{0,50}(hair|buhok|floor|sahig)",
            text_lower,
        ):
            boost += 25.0

        # Violence against a pregnant victim greatly increases risk.
        if re.search(
            r"(pregnant|buntis).{0,50}(kick|kicked|sinipa|hit|hampas|suntok|tinulak|push)|"
            r"(kick|kicked|sinipa|hit|hampas|suntok|tinulak|push).{0,50}(pregnant|buntis)",
            text_lower,
        ):
            boost += 25.0

        return boost
    
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
        text_normalized = self._normalize_noisy_text(text_lower)
        risk_score = 0.0
        nonviolent_saksak = self._is_nonviolent_saksak_context(text_lower)
        nonviolent_kill = self._is_nonviolent_kill_context(text_lower)
        nonviolent_shoot = self._is_nonviolent_shoot_context(text_lower)
        nonviolent_impact = self._is_nonviolent_impact_context(text_lower)
        
        # Check critical keywords
        for keyword, weight in self.CRITICAL_KEYWORDS.items():
            if nonviolent_saksak and keyword in self.SAKSAK_TERMS:
                continue
            if nonviolent_kill and keyword in {"kill", "killed", "killing", "kill you"}:
                continue
            if nonviolent_shoot and keyword in self.SHOOT_TERMS:
                continue
            count = self._keyword_count(text_lower, keyword, normalized_text=text_normalized)
            risk_score += count * weight
        
        # Check high-risk keywords
        for keyword, weight in self.HIGH_RISK_KEYWORDS.items():
            if nonviolent_impact and keyword in {"beat", "beating", "hit", "punch"}:
                continue
            count = self._keyword_count(text_lower, keyword, normalized_text=text_normalized)
            risk_score += count * weight
        
        # Check medium-risk keywords
        for keyword, weight in self.MEDIUM_RISK_KEYWORDS.items():
            count = self._keyword_count(text_lower, keyword, normalized_text=text_normalized)
            risk_score += count * weight
        
        # Check Tagalog keywords
        for keyword, weight in self.TAGALOG_CRITICAL.items():
            if nonviolent_saksak and keyword in {"sinaksak", "saksak", "tinaga"}:
                continue
            if nonviolent_kill and keyword in {"patay", "patayin", "papatayin"}:
                continue
            if nonviolent_shoot and keyword in {"binaril", "barilin", "pinagbabaril"}:
                continue
            count = self._keyword_count(text_lower, keyword, normalized_text=text_normalized)
            risk_score += count * weight
        
        for keyword, weight in self.TAGALOG_HIGH.items():
            if nonviolent_impact and keyword in {"suntok", "sinuntok", "sinusuntok"}:
                continue
            count = self._keyword_count(text_lower, keyword, normalized_text=text_normalized)
            risk_score += count * weight

        # Add pattern-based severity boosts for dangerous physical contexts.
        if not (nonviolent_saksak or nonviolent_kill or nonviolent_shoot):
            risk_score += self._severe_risk_boost(text_lower)

        # Strong extra boost when a child/baby is the victim of violence.
        risk_score += self._child_victim_violence_boost(text_lower)
        
        # Normalize to 0-100 range
        # Base risk should be 0 so negative/irrelevant reports stay near 0.
        base_risk = 0.0
        
        # Cap maximum score contribution
        max_contribution = 95.0
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
        # User-defined policy:
        # - P3: 40 below
        # - P2: 41 to 69
        # - P1: 70+
        if risk_percentage >= 70:
            return 'First Priority (P1)'
        elif risk_percentage > 40:
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

        # Extra floor for physical abuse involving children
        if incident_type == "Physical Abuse" and children_involved:
            risk = max(risk, 72.0)

        # Clamp to 0-100
        risk = max(0.0, min(risk, 100.0))
        return round(risk, 2)