# nlp_utils.py

"""
Small helper functions for working with lab report text.

Step 1: clean_report_text  -> makes the text neat (remove blank lines, extra spaces)
Step 2: extract_parameters -> tries to read lines like:
    Hemoglobin: 12.3 g/dL
    Hb = 9.1 g/dL
    Glucose  152 mg/dL (H)
and convert them into Python dictionaries.
"""

import re
from typing import List, Dict


def clean_report_text(text: str) -> str:
    """
    Clean the raw lab report text.

    - Make all line endings consistent
    - Remove extra spaces at the start/end of each line
    - Remove empty lines

    This makes later processing more reliable.
    """
    # Convert Windows/Mac line endings to simple "\n"
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into lines
    lines = text.split("\n")

    # Strip spaces and drop empty lines
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:  # ignore completely empty lines
            cleaned_lines.append(stripped)

    # Join back into a cleaned string
    return "\n".join(cleaned_lines)


def extract_parameters(text: str) -> List[Dict]:
    """
    Extract simple lab parameters from cleaned text using a regular expression.

    We expect lines like:
      Hemoglobin: 12.3 g/dL
      Hb = 9.1 g/dL
      Glucose  152 mg/dL (H)

    For each matching line we return a dictionary:
    {
        "parameter": "Hemoglobin",
        "reported_value": "12.3",
        "unit": "g/dL",
        "raw_line": "Hemoglobin: 12.3 g/dL"
    }

    This is a very simple extractor, good enough for the assignment.
    """

    results: List[Dict] = []

    # Regular expression:
    # - name: letters/numbers/spaces etc at the start
    # - optional ":" or "=" after the name
    # - numeric value
    # - optional unit
    # - optional flag in parentheses (like (H), (L)) which we ignore
    pattern = re.compile(
        r"""^
        (?P<name>[A-Za-z0-9\s\-\(\)\/]+?)   # parameter name
        [:\=]?\s*                            # optional : or = with spaces
        (?P<value>-?\d+(\.\d+)?)             # numeric value (int or float)
        \s*
        (?P<unit>[A-Za-z/%µ\.\d\^\s\*x\+]*?)  # unit: g/dL, %, x10^9/L, etc.
        (\s*\(.*\))?                         # optional flag in parentheses
        $""",
        re.VERBOSE,
    )

    # Check each line separately
    for line in text.split("\n"):
        match = pattern.match(line.strip())
        if not match:
            continue  # line does not match the pattern

        name = match.group("name").strip()
        value = match.group("value").strip()
        unit = match.group("unit").strip() or None

        results.append(
            {
                "parameter": name,
                "reported_value": value,
                "unit": unit,
                "raw_line": line.strip(),
            }
        )

    return results