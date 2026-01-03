

OIL_DETECTION_SOLUTIONS = {
    "healthy": {
        "issue_name": "Intact Structure",
        "severity": "none",
        "description": "The inspected area appears to be in good condition with no signs of leakage or corrosion.",
        "mitigation": "Continue regular monitoring and scheduled maintenance.",
        "prevention": [
            "Maintain protective coatings",
            "Regular visual inspections",
            "Environmental monitoring",
            "Periodic pressure testing"
        ],
        "symptoms": []
    },
    "oil_leakage": {
        "issue_name": "Oil Leakage",
        "severity": "high",
        "description": "Active discharge of oil from a pipe, valve, or storage unit. May lead to environmental impact or fire hazard.",
        "symptoms": [
            "Visible fluid discharge",
            "Pressure drop in lines",
            "Surface staining",
            "Odor of hydrocarbons"
        ],
        "mitigation": "Immediate containment. Isolate the affected section, deploy absorbent booms, and repair the breach.",
        "prevention": [
            "Regular valve testing",
            "Seal integrity checks",
            "Smart pigging for internal pipe inspection",
            "Vibration monitoring"
        ],
        "action_plan": [
            "Activate emergency shutdown (ESD)",
            "Notify environmental safety teams",
            "Deploy spill kits",
            "Repair or replace leaking component"
        ],
        "prognosis": "Variable depending on response time and leakage volume."
    },
    "corrosion": {
        "issue_name": "Surface Corrosion",
        "severity": "medium",
        "description": "Chemical or electrochemical reaction between the metal and its environment, leading to material degradation.",
        "symptoms": [
            "Rust or pitting on surfaces",
            "Flaking paint or coating",
            "Discoloration (brown/orange)",
            "Wall thickness reduction"
        ],
        "mitigation": "Clean affected area, measure wall loss, and apply protective anti-corrosion coating or sacrificial anodes.",
        "prevention": [
            "Apply epoxy or polyurethane coatings",
            "Cathodic protection systems",
            "Use corrosion inhibitors",
            "Regular ultrasonic thickness measurements"
        ],
        "action_plan": [
            "Mechanical cleaning (sandblasting)",
            "Wall thickness inspection (NDT)",
            "Repainting/Coating application",
            "Reinforcement if required"
        ],
        "prognosis": "Usually manageable if detected early; critical if structural integrity is compromised."
    },
    "severe_corrosion": {
        "issue_name": "Severe Corrosion/Structural Damage",
        "severity": "very high",
        "description": "Advanced material loss that significantly weakens structural integrity. High risk of catastrophic failure.",
        "symptoms": [
            "Deep pits or holes",
            "Significant metal loss",
            "Structural deformation",
            "Leaking at corroded points"
        ],
        "mitigation": "Immediate reinforcement or replacement of the affected section. Potential localized shutdown.",
        "prevention": [
            "Aggressive corrosion monitoring",
            "Regular NDT inspections",
            "Upgrade material selection (e.g., stainless steel)",
            "Improved drainage to avoid water collection"
        ],
        "action_plan": [
            "Structural integrity assessment",
            "Emergency repair clamps or sleeves",
            "Section replacement",
            "Root cause analysis"
        ],
        "prognosis": "Critical. Immediate intervention required to prevent failure."
    }
}

def get_oil_solution(issue_key: str) -> dict:
    """
    Get treatment solution for a detected oil/corrosion issue
    
    Args:
        issue_key: Issue identifier (e.g., 'oil_leakage', 'corrosion')
    
    Returns:
        Dictionary containing issue information and mitigation
    """
    issue_key = issue_key.lower().replace(" ", "_").replace("-", "_")
    return OIL_DETECTION_SOLUTIONS.get(issue_key, {
        "issue_name": "Unknown Issue",
        "severity": "unknown",
        "description": "Issue not recognized in database",
        "mitigation": "Consult with a mechanical integrity engineer or safety supervisor",
        "prevention": ["Enhanced monitoring and site inspection"],
        "prognosis": "Unknown - professional assessment recommended"
    })

def get_all_issues() -> list:
    """Get list of all issues in the database"""
    return list(OIL_DETECTION_SOLUTIONS.keys())

def get_severity_color(severity: str) -> str:
    """Get color code for severity level"""
    colors = {
        "none": "#28a745",      # green
        "low": "#ffc107",       # yellow
        "medium": "#fd7e14",    # orange
        "high": "#dc3545",      # red
        "very high": "#6f0000", # dark red
        "unknown": "#6c757d"    # gray
    }
    return colors.get(severity.lower(), "#6c757d")
