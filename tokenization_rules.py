import logging

class TokenizationRules:
    _RULE_TEMPLATES = {
        ('HR', 'SALES'): {
            'tokenize': ['Name', 'Phone', 'Salary', 'Email', 'Department'],
            'pass_through': ['Position']
        },
        ('HR', 'FINANCE'): {
            'tokenize': ['Name', 'Phone', 'Email', 'Salary', 'Department'],
            'pass_through': ['Position']
        },
        ('HR', 'IT'): {
            'tokenize': ['Name', 'Salary', 'Phone', 'Department'],
            'pass_through': ['Email', 'Position']
        },
        ('SALES', 'FINANCE'): {
            'tokenize': ['Name', 'Email', 'Phone', 'Salary', 'Department'],
            'pass_through': ['Position']
        },
        ('SALES', 'IT'): {
            'tokenize': ['Name', 'Salary', 'Phone', 'Department'],
            'pass_through': ['Email', 'Position']
        },
        ('FINANCE', 'IT'): {
            'tokenize': ['Name', 'Phone', 'Email', 'Salary', 'Department'],
            'pass_through': ['Position']
        }
    }

    DEPARTMENT_RULES = {}
    for (src, dst), rules in _RULE_TEMPLATES.items():
        DEPARTMENT_RULES[(src, dst)] = rules
        DEPARTMENT_RULES[(dst, src)] = rules

    def __init__(self):
        self.current_source = None
        self.current_dest = None
        self.current_rules = None

    def should_tokenize(self, field):
        """Check if a field should be tokenized based on current rules."""
        if not self.current_rules:
            logging.warning("No rules currently set")
            return False
        
        return field in self.current_rules.get('tokenize', [])

    def set_departments(self, source_dept, dest_dept):
        """Set the current source and destination departments."""
        self.current_source = source_dept.upper()
        self.current_dest = dest_dept.upper()
        self.current_rules = self.get_rules(self.current_source, self.current_dest)
        return bool(self.current_rules)

    @classmethod
    def get_rules(cls, source_dept, dest_dept):
        """Get tokenization rules for a specific department pair."""
        source_dept = source_dept.upper()
        dest_dept = dest_dept.upper()
        
        logging.info(f"Looking up rules for {source_dept} -> {dest_dept}")
        
        key = (source_dept, dest_dept)
        if key not in cls.DEPARTMENT_RULES:
            logging.error(f"No rules found for {key}. Available rules: {list(cls.DEPARTMENT_RULES.keys())}")
            return None
        
        rules = cls.DEPARTMENT_RULES[key]
        logging.info(f"Found rules: tokenize={rules['tokenize']}, pass_through={rules['pass_through']}")
        return rules
        
    @classmethod
    def parse_filename(cls, filename):
        """Parse source and destination departments from filename.
        Handles formats: HR_to_SALES.csv, 20250307_052347_HR_to_IT.csv, processed_HR_to_SALES.csv
        """
        try:
            name = filename.replace('.csv', '')
            if name.startswith('processed_'):
                name = name[len('processed_'):]
            parts = name.split('_to_')
            if len(parts) < 2:
                logging.error(f"Invalid filename format: {filename}")
                return None, None
            parts = parts[-2:]
            parts_before_to = parts[0].rsplit('_', 1) if '_' in parts[0] else [parts[0]]
            if len(parts_before_to) > 1 and parts_before_to[0].isdigit():
                source_dept = parts_before_to[1]
            else:
                source_dept = parts_before_to[-1]
            dest_dept = parts[1].split('_')[0]
            return source_dept.upper(), dest_dept.upper()
        except Exception as e:
            logging.error(f"Error parsing filename {filename}: {str(e)}")
            return None, None
