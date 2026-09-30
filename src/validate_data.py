import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_bank_schema():
    """Defines the strict schema specification for the Bank Marketing dataset."""
    yes_no_unknown = ["yes", "no", "unknown"]
    return pa.DataFrameSchema({
        "age": Column(pa.Int, Check.in_range(17, 100)),
        "job": Column(pa.String, Check.isin([
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"
        ])),
        "marital": Column(pa.String, Check.isin(["divorced", "married", "single", "unknown"])),
        "education": Column(pa.String, Check.isin([
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown"
        ])),
        "default": Column(pa.String, Check.isin(yes_no_unknown)),
        "housing": Column(pa.String, Check.isin(yes_no_unknown)),
        "loan": Column(pa.String, Check.isin(yes_no_unknown)),
        "contact": Column(pa.String, Check.isin(["cellular", "telephone"])),
        "month": Column(pa.String, Check.isin([
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ])),
        "day_of_week": Column(pa.String, Check.isin(["mon", "tue", "wed", "thu", "fri"])),
        "duration": Column(pa.Int, Check.ge(0)),
        "campaign": Column(pa.Int, Check.ge(1)),
        "pdays": Column(pa.Int, Check.in_range(0, 999)),   # 999 = not previously contacted
        "previous": Column(pa.Int, Check.ge(0)),
        "poutcome": Column(pa.String, Check.isin(["failure", "nonexistent", "success"])),
        "emp.var.rate": Column(pa.Float, Check.in_range(-5.0, 5.0)),
        "cons.price.idx": Column(pa.Float, Check.in_range(90.0, 95.0)),
        "cons.conf.idx": Column(pa.Float, Check.in_range(-55.0, -20.0)),
        "euribor3m": Column(pa.Float, Check.in_range(0.0, 6.0)),
        "nr.employed": Column(pa.Float, Check.in_range(4900.0, 5300.0)),
        "y": Column(pa.String, Check.isin(["yes", "no"]))
    }, strict=True)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_bank_schema()
    
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())
        
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False

if __name__ == "__main__":
    # When run directly, it only validates the clean production data
    data_path = 'data/raw/bank-additional-full.csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path, sep=';')
        ok = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        # Non-zero exit code so pipelines/run_lab5_pipeline.py halts on bad data
        raise SystemExit(0 if ok else 1)
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
        raise SystemExit(1)
