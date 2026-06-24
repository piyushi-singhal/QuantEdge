"""
Risk Scoring Engine — QuantEdge

Determines sensitivity risk scores for data fields based on:
1. Field name heuristics (column header analysis)
2. Value pattern analysis (regex rules)
3. Content entropy calculation

Risk Score Range: 0 (not sensitive) — 100 (maximum sensitivity)
"""
import re
import math
from typing import Dict, Any, Optional


class RiskScorer:
    # Field-level base scores by column name
    FIELD_BASE_SCORES = {
        # Identity fields
        'ssn': 98,
        'social_security': 98,
        'national_id': 97,
        'passport': 96,
        'drivers_license': 90,
        'driver_license': 90,
        'license_number': 85,
        # Financial fields
        'credit_card': 95,
        'creditcard': 95,
        'cc_number': 95,
        'card_number': 95,
        'bank_account': 94,
        'routing_number': 88,
        'salary': 92,
        'compensation': 90,
        'wage': 85,
        'bonus': 85,
        'income': 85,
        # Contact fields
        'email': 75,
        'email_address': 75,
        'phone': 70,
        'phone_number': 70,
        'mobile': 70,
        'telephone': 68,
        'address': 65,
        'street': 60,
        'city': 30,
        'state': 25,
        'zip': 40,
        'zip_code': 40,
        'postal': 40,
        # Employment fields
        'name': 80,
        'full_name': 80,
        'first_name': 75,
        'last_name': 75,
        'employee_id': 70,
        'department': 20,
        'position': 15,
        'title': 10,
        'role': 10,
        # Healthcare fields
        'diagnosis': 95,
        'medical': 95,
        'health': 90,
        'blood_type': 70,
        'bloodtype': 70,
        'patient_id': 85,
        # Demographics
        'dob': 90,
        'date_of_birth': 90,
        'birth_date': 90,
        'age': 30,
        'gender': 20,
        'race': 60,
        'ethnicity': 60,
        # System fields
        'password': 100,
        'secret': 100,
        'token': 50,
        'key': 90,
        'api_key': 95,
    }

    # Value pattern rules with risk contribution
    PATTERN_RULES = [
        (r'^\d{3}-\d{2}-\d{4}$', 25, 'SSN format'),
        (r'^\d{3}-?\d{2}-?\d{4}$', 25, 'SSN-like'),
        (r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$', 20, 'Credit card format'),
        (r'^[\w\.-]+@[\w\.-]+\.\w+$', 15, 'Email format'),
        (r'^\+?1?\d{9,15}$', 10, 'Phone format'),
        (r'^[A-Z]{2}\d{6}$', 15, 'Passport-like'),
        (r'^\d{9}$', 10, 'US SSN-like (9 digits)'),
        (r'^\$\s*[\d,]+\.?\d*$', 15, 'Currency amount'),
        (r'^\d{5}(-\d{4})?$', 5, 'ZIP code'),
    ]

    # Per-department risk modifiers
    DEPARTMENT_MODIFIERS = {
        'HR': 1.15,
        'FINANCE': 1.10,
        'SALES': 0.90,
        'IT': 0.85,
        'ENGINEERING': 0.80,
        'MARKETING': 0.75,
        'LEGAL': 1.20,
    }

    # Expiry durations in minutes by risk tier
    EXPIRY_TIERS = [
        (80, 5, 'High', 'immediate'),
        (50, 15, 'Medium', 'short'),
        (0, 30, 'Low', 'standard'),
    ]

    @classmethod
    def score_field_by_name(cls, field_name: str) -> tuple:
        """Score a field based on its column name.
        Returns (base_score, match_type).
        """
        normalized = field_name.lower().strip().replace(' ', '_').replace('-', '_')
        for pattern, score in cls.FIELD_BASE_SCORES.items():
            if normalized == pattern or normalized.endswith('_' + pattern) or normalized.startswith(pattern + '_'):
                return (score, 'exact_field_match')
            if pattern in normalized:
                return (max(score - 10, 0), 'partial_field_match')
        return (50, 'unknown_field')  # Default medium risk for unknown fields

    @classmethod
    def score_value_pattern(cls, value: str) -> int:
        """Add risk contribution from value pattern matching."""
        for pattern, contribution, _ in cls.PATTERN_RULES:
            if re.match(pattern, str(value)):
                return contribution
        return 0

    @classmethod
    def calculate_entropy(cls, value: str) -> float:
        """Calculate Shannon entropy of a string value.
        Higher entropy = more random-looking = potentially more sensitive.
        """
        if not value:
            return 0.0
        value = str(value)
        prob = [float(value.count(c)) / len(value) for c in set(value)]
        entropy = -sum(p * math.log2(p) for p in prob)
        return round(entropy, 2)

    @classmethod
    def entropy_risk_modifier(cls, entropy: float) -> int:
        """Convert entropy to risk modifier (-10 to +15)."""
        if entropy > 4.0:
            return 15  # High entropy suggests encoded/encrypted data
        elif entropy > 3.0:
            return 5
        elif entropy < 1.5:
            return -10  # Low entropy suggests predictable values
        return 0

    @classmethod
    def compute_risk(cls, field_name: str, value: Any, source_dept: str = None) -> dict:
        """Compute comprehensive risk score for a field-value pair.
        
        Returns:
            dict with keys: score, tier, expiry_minutes, entropy, 
                            field_score, pattern_contribution, entropy_modifier
        """
        field_score, match_type = cls.score_field_by_name(field_name)
        pattern_contrib = cls.score_value_pattern(str(value) if value else '')
        entropy = cls.calculate_entropy(str(value) if value else '')
        entropy_mod = cls.entropy_risk_modifier(entropy)

        # Department modifier
        dept_mod = cls.DEPARTMENT_MODIFIERS.get(source_dept.upper() if source_dept else '', 1.0)

        # Final score: field baseline + pattern contribution + entropy modifier, scaled by dept
        raw_score = (field_score + pattern_contrib + entropy_mod) * dept_mod
        final_score = min(max(round(raw_score), 0), 100)

        # Determine tier and expiry
        tier_name = 'Low'
        expiry_min = 30
        for threshold, mins, t_name, _ in cls.EXPIRY_TIERS:
            if final_score >= threshold:
                tier_name = t_name
                expiry_min = mins
                break

        return {
            'score': final_score,
            'tier': tier_name,
            'expiry_minutes': expiry_min,
            'entropy': entropy,
            'field_score': field_score,
            'pattern_contribution': pattern_contrib,
            'entropy_modifier': entropy_mod,
            'department_modifier': round(dept_mod, 2),
            'match_type': match_type,
        }

    @classmethod
    def risk_display(cls, score: int) -> str:
        """Return CSS class or color for risk score display."""
        if score >= 80:
            return 'critical'
        elif score >= 50:
            return 'high'
        elif score >= 25:
            return 'medium'
        return 'low'
