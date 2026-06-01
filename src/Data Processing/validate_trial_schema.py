import json
from pathlib import Path


def type_matches(val, expected_types):
    if not isinstance(expected_types, list):
        expected_types = [expected_types]

    for t in expected_types:
        if t == "null" and val is None:
            return True
        if t == "float" and isinstance(val, (int, float)):
            return True
        if t == "integer" and isinstance(val, int):
            return True
        if t == "boolean" and isinstance(val, bool):
            return True
        if t == "string" and isinstance(val, str):
            return True
        if t == "categorical" and isinstance(val, str):
            return True

    return False


def validate_field(value, rules, field_path):
    errors = []

    # Type check
    if not type_matches(value, rules["type"]):
        errors.append(f"{field_path}: type mismatch (value={value})")
        return errors

    # Null allowed
    if value is None:
        return errors

    # Value constraint
    if "values" in rules and value not in rules["values"]:
        errors.append(f"{field_path}: value '{value}' not in {rules['values']}")

    # Range constraint
    if "range" in rules:
        lo, hi = rules["range"]
        if not isinstance(value, (int, float)) or not (lo <= value <= hi):
            errors.append(f"{field_path}: value {value} out of range [{lo}, {hi}]")

    # Min constraint
    if "min" in rules:
        if not isinstance(value, (int, float)) or value < rules["min"]:
            errors.append(f"{field_path}: value {value} < min {rules['min']}")

    return errors


def validate_row(row, schema):
    errors = []

    for group, fields in schema.items():
        if group in {
            "schema_name",
            "schema_version",
            "unit_of_analysis",
            "description",
            "notes"
        }:
            continue

        if group not in row:
            errors.append(f"Missing group: {group}")
            continue

        group_data = row[group]

        for field_name, rules in fields.items():
            field_path = f"{group}.{field_name}"

            if field_name not in group_data:
                errors.append(f"Missing field: {field_path}")
                continue

            value = group_data[field_name]
            errors.extend(validate_field(value, rules, field_path))

    return errors


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    schema_path = project_root / "schemas" / "trial_feature_schema_v1.json"
    row_path = project_root / "example_trial_row.json"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(row_path, "r", encoding="utf-8") as f:
        row = json.load(f)

    errors = validate_row(row, schema)

    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors))

    print("✅ Schema validation passed.")
