"""
Language Detection Module
Detects language in incident descriptions (English, Tagalog, Ilocano, Pangasinan, Mixed)
"""

import re
from typing import Dict, List, Tuple
from langdetect import detect, detect_langs, LangDetectException


class LanguageDetector:
    """Detects language in incident reports"""
    
    # Language keywords for better detection
    TAGALOG_KEYWORDS = [
        'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila',
        'ng', 'na', 'sa', 'ang', 'mga', 'ay', 'at', 'o',
        'hindi', 'oo', 'hindi', 'po', 'opo', 'hindi po',
        'kumusta', 'salamat', 'magandang', 'umaga', 'hapon', 'gabi'
    ]
    
    ILOCANO_KEYWORDS = [
        'siak', 'sika', 'isu', 'dakami', 'dakayo', 'isuda',
        'ti', 'nga', 'iti', 'dagiti', 'ket', 'wenno',
        'saan', 'wen', 'apo', 'kumusta', 'agyamanak'
    ]
    
    PANGASINAN_KEYWORDS = [
        'siak', 'sika', 'sikato', 'kami', 'kayo', 'sikara',
        'na', 'ed', 'so', 'saray', 'tan', 'odino',
        'andi', 'on', 'apo'
    ]
    
    def __init__(self):
        """Initialize language detector"""
        pass
    
    def detect_language(self, text: str) -> Dict[str, any]:
        """
        Detect language in text
        
        Returns:
            Dict with 'language' (primary) and 'confidence' (score)
        """
        if not text or not text.strip():
            return {'language': 'English', 'confidence': 0.0}
        
        text_lower = text.lower()
        
        # Count keyword matches
        tagalog_count = sum(1 for kw in self.TAGALOG_KEYWORDS if kw in text_lower)
        ilocano_count = sum(1 for kw in self.ILOCANO_KEYWORDS if kw in text_lower)
        pangasinan_count = sum(1 for kw in self.PANGASINAN_KEYWORDS if kw in text_lower)
        
        # Use langdetect for primary detection
        try:
            detected_langs = detect_langs(text)
            primary_lang = detected_langs[0].lang if detected_langs else 'en'
            primary_confidence = detected_langs[0].prob if detected_langs else 0.0
        except LangDetectException:
            primary_lang = 'en'
            primary_confidence = 0.5
        
        # Determine language based on keywords and langdetect
        language_scores = {
            'English': 0.0,
            'Tagalog': tagalog_count * 0.1,
            'Ilocano': ilocano_count * 0.1,
            'Pangasinan': pangasinan_count * 0.1,
            'Mixed Language': 0.0
        }
        
        # Boost score based on langdetect
        if primary_lang == 'tl':  # Tagalog
            language_scores['Tagalog'] += primary_confidence * 0.5
        elif primary_lang == 'en':
            language_scores['English'] += primary_confidence * 0.5
        
        # Check for mixed language (multiple languages detected)
        if len(detected_langs) > 1:
            second_lang_prob = detected_langs[1].prob if len(detected_langs) > 1 else 0.0
            if second_lang_prob > 0.2:  # Significant second language
                language_scores['Mixed Language'] = 0.6
        
        # If multiple keyword matches, likely mixed
        keyword_matches = sum([
            tagalog_count > 2,
            ilocano_count > 2,
            pangasinan_count > 2
        ])
        
        if keyword_matches > 1:
            language_scores['Mixed Language'] = max(language_scores['Mixed Language'], 0.5)
        
        # Get highest scoring language
        detected_language = max(language_scores.items(), key=lambda x: x[1])[0]
        confidence = language_scores[detected_language]
        
        # Normalize confidence
        confidence = min(confidence, 1.0)
        if confidence < 0.3:
            detected_language = 'English'  # Default fallback
            confidence = 0.5
        
        return {
            'language': detected_language,
            'confidence': confidence
        }
