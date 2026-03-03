# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion
# MAGIC
# MAGIC Ingest sample corporate documents into a Unity Catalog Delta table.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Generate sample corporate documents (HR, IT, Finance, General)
# MAGIC 2. Chunk documents for vector search
# MAGIC 3. Save to Delta table in Unity Catalog

# COMMAND ----------

# --- Configuration (values come from DAB job parameters -> databricks.yml variables) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
TABLE_NAME = "doc_chunks"

print(f"Target table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Create Schema

# COMMAND ----------

# The schema is created by the DAB bundle (resources.schemas.corporate_schema).
# We only ensure the catalog exists as a safety net.
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Generate Sample Corporate Documents
# MAGIC
# MAGIC In a real scenario, you would load documents from Unity Catalog Volumes
# MAGIC or cloud storage. Here we create sample docs for the demo.

# COMMAND ----------

sample_documents = [
    {
        "source_file": "employee_handbook.pdf",
        "department": "HR",
        "content": """
Employee Handbook - Remote Work Policy (Effective: January 2025)

Section 3.1: Eligibility
All full-time employees who have completed their probation period (90 days) are eligible
for remote work arrangements. Contractors and temporary employees should consult with
their hiring manager for remote work options.

Section 3.2: Remote Work Schedule
Employees in eligible roles may work remotely up to 3 days per week. The specific days
must be agreed upon with the employee's direct manager. Core collaboration hours are
10 AM - 3 PM local time, during which all employees must be available.

Section 3.3: Equipment Stipend
A one-time equipment stipend of $500 is available for home office setup. This covers
ergonomic chair, monitor, keyboard, or other approved office equipment. Submit requests
through the Concur expense system within 30 days of approval.

Section 3.4: Internet Reimbursement
The company reimburses up to $75/month for home internet costs. Employees must maintain
a minimum 50 Mbps connection for video conferencing.
""",
    },
    {
        "source_file": "parental_leave_policy.pdf",
        "department": "HR",
        "content": """
Parental Leave Policy (Effective: March 2025)

Section 1: Overview
The company is committed to supporting employees during major life events. Our parental
leave policy applies to all full-time employees regardless of gender or family structure.

Section 2: Leave Duration
- Primary caregiver: 16 weeks of paid leave at 100% salary
- Secondary caregiver: 8 weeks of paid leave at 100% salary
- Adoption/foster care: Same benefits as biological parents

Section 3: Eligibility
Employees must have been employed for at least 12 months and worked 1,250 hours in
the preceding 12 months.

Section 4: Return to Work
Employees returning from parental leave are guaranteed the same or equivalent position.
A phased return schedule (4 weeks at 80% capacity) is available upon request.
""",
    },
    {
        "source_file": "expense_policy.pdf",
        "department": "Finance",
        "content": """
Expense Reimbursement Policy (Effective: January 2025)

Section 1: Expense Reports
All expense reports must be submitted through the Concur system within 30 days of
the expense. Receipts are required for all expenses over $25.

Section 2: Travel
- Flights: Must be booked through the corporate travel portal (Concur Travel)
- Hotels: Maximum nightly rate of $250 for domestic, $350 for international
- Meals: $75/day per diem for domestic travel, $100/day for international
- Ground transportation: Rideshare preferred; rental cars require manager approval

Section 3: Client Entertainment
- Pre-approval required for expenses over $200
- Maximum $150/person for client dinners
- All client entertainment must include business purpose documentation

Section 4: Approval Workflow
- Under $500: Direct manager approval
- $500-$2,000: Director approval
- Over $2,000: VP approval required
""",
    },
    {
        "source_file": "it_security_guide.pdf",
        "department": "IT",
        "content": """
IT Security Guidelines (Effective: February 2025)

Section 1: Password Policy
- Minimum 12 characters with uppercase, lowercase, numbers, and symbols
- Passwords must be changed every 90 days
- Multi-factor authentication (MFA) is required for all corporate applications
- Use the company password manager (1Password) for storing credentials

Section 2: Device Security
- All company laptops must have full-disk encryption enabled
- Automatic screen lock after 5 minutes of inactivity
- Personal devices accessing company data must be enrolled in MDM
- Report lost or stolen devices to IT within 1 hour

Section 3: Data Classification
- Public: Marketing materials, press releases
- Internal: Project plans, meeting notes
- Confidential: Financial data, HR records, customer PII
- Restricted: Trade secrets, M&A documents

Section 4: Incident Reporting
Report security incidents to security@company.com or call the IT helpdesk at ext. 5555.
All incidents must be reported within 4 hours of discovery.
""",
    },
    {
        "source_file": "company_holidays_2025.pdf",
        "department": "General",
        "content": """
Company Holidays 2025

The company observes the following paid holidays:
- New Year's Day: January 1
- Martin Luther King Jr. Day: January 20
- Presidents' Day: February 17
- Memorial Day: May 26
- Independence Day: July 4
- Labor Day: September 1
- Thanksgiving: November 27-28 (2 days)
- Christmas Eve & Christmas Day: December 24-25
- Winter Break: December 26-31 (company-wide shutdown)

Total: 13 paid holiday days

Additionally, employees receive 2 floating holidays per year to use at their discretion.
Floating holidays do not roll over and must be used within the calendar year.
""",
    },
]

print(f"Generated {len(sample_documents)} sample documents")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Chunk Documents

# COMMAND ----------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Simple text chunking by character count with overlap."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


# Chunk all documents
all_chunks = []
for doc in sample_documents:
    chunks = chunk_text(doc["content"], chunk_size=800, overlap=100)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{doc['source_file']}_{i}",
            "content": chunk,
            "source_file": doc["source_file"],
            "department": doc["department"],
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

print(f"Created {len(all_chunks)} chunks from {len(sample_documents)} documents")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Save to Delta Table

# COMMAND ----------

import pandas as pd

df = spark.createDataFrame(pd.DataFrame(all_chunks))

table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# Enable Change Data Feed BEFORE writing data so Vector Search Delta Sync
# can capture the initial load. Table properties persist across overwrites.
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table_path} (
        chunk_id STRING, content STRING, source_file STRING,
        department STRING, chunk_index INT, total_chunks INT
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_path)

count = spark.table(table_path).count()
print(f"Saved {count} chunks to {table_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Verify

# COMMAND ----------

display(spark.table(table_path).limit(5))

# COMMAND ----------

print(f"""
Data Ingestion Complete!
========================
Table: {table_path}
Chunks: {count}
Documents: {len(sample_documents)}

Next: Run notebook 02_vector_index_creation.py
""")
