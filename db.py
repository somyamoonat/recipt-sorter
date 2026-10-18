"""
db.py - SQLite Database Management for ReceiptSorter
Handles persistent local storage, migrations, querying, duplicates, and analytics.
Everything runs locally on sqlite3 in data/receipts.db.
"""

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "receipts.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates directory if needed and returns an active SQLite connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the receipts table and indexes if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                vendor TEXT NOT NULL,
                date TEXT,
                total_amount REAL NOT NULL DEFAULT 0.0,
                tax_amount REAL NOT NULL DEFAULT 0.0,
                currency TEXT DEFAULT '',
                category TEXT DEFAULT 'Other',
                client_project TEXT DEFAULT '',
                tax_deductible INTEGER NOT NULL DEFAULT 0,
                line_items_summary TEXT DEFAULT '',
                confidence TEXT DEFAULT 'low',
                notes TEXT DEFAULT '',
                image_hash TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        # Create helpful query indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_vendor ON receipts(vendor);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_cat ON receipts(category);")
        conn.commit()


def check_duplicate(
    vendor: str,
    date: str,
    total_amount: float,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Finds existing receipts in the database with matching vendor, date, and total_amount.
    Used for duplicate detection before saving.
    """
    clean_vendor = (vendor or "").strip().lower()
    clean_date = (date or "").strip()
    round_amount = round(float(total_amount or 0.0), 2)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, file_name, vendor, date, total_amount, currency, category, created_at
            FROM receipts
            WHERE LOWER(TRIM(vendor)) = ?
              AND TRIM(date) = ?
              AND ROUND(total_amount, 2) = ?
            """,
            (clean_vendor, clean_date, round_amount),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def save_receipt(data: Dict[str, Any], db_path: str = DB_PATH) -> int:
    """Saves a single receipt to SQLite and returns its new ID."""
    now_iso = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO receipts (
                file_name, vendor, date, total_amount, tax_amount,
                currency, category, client_project, tax_deductible,
                line_items_summary, confidence, notes, image_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("file_name", ""),
                data.get("vendor", "NEEDS REVIEW"),
                data.get("date", ""),
                float(data.get("total_amount", 0.0)),
                float(data.get("tax_amount", 0.0)),
                (data.get("currency") or "").upper()[:3],
                data.get("category", "Other"),
                data.get("client_project", ""),
                1 if data.get("tax_deductible") in [True, 1, "1", "true"] else 0,
                data.get("line_items_summary", "")[:80],
                data.get("confidence", "low"),
                data.get("notes", ""),
                data.get("image_hash", ""),
                now_iso,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def save_receipts_batch(receipts_list: List[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """Saves multiple receipts in a single atomic transaction."""
    saved_count = 0
    now_iso = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for r in receipts_list:
            cursor.execute(
                """
                INSERT INTO receipts (
                    file_name, vendor, date, total_amount, tax_amount,
                    currency, category, client_project, tax_deductible,
                    line_items_summary, confidence, notes, image_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("file_name", ""),
                    r.get("vendor", "NEEDS REVIEW"),
                    r.get("date", ""),
                    float(r.get("total_amount", 0.0)),
                    float(r.get("tax_amount", 0.0)),
                    (r.get("currency") or "").upper()[:3],
                    r.get("category", "Other"),
                    r.get("client_project", ""),
                    1 if r.get("tax_deductible") in [True, 1, "1", "true"] else 0,
                    r.get("line_items_summary", "")[:80],
                    r.get("confidence", "low"),
                    r.get("notes", ""),
                    r.get("image_hash", ""),
                    now_iso,
                ),
            )
            saved_count += 1
        conn.commit()
    return saved_count


def get_all_receipts(
    search: str = "",
    category: Optional[str] = None,
    client_project: Optional[str] = None,
    currency: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tax_deductible_only: bool = False,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Queries receipts with flexible search and filtering.
    Returns a pandas DataFrame sorted by date descending.
    """
    query = "SELECT * FROM receipts WHERE 1=1"
    params: List[Any] = []

    if search:
        search_like = f"%{search.strip().lower()}%"
        query += " AND (LOWER(vendor) LIKE ? OR LOWER(notes) LIKE ? OR LOWER(client_project) LIKE ? OR LOWER(line_items_summary) LIKE ?)"
        params.extend([search_like, search_like, search_like, search_like])

    if category and category != "All":
        query += " AND category = ?"
        params.append(category)

    if client_project and client_project != "All":
        query += " AND client_project = ?"
        params.append(client_project)

    if currency and currency != "All":
        query += " AND currency = ?"
        params.append(currency)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    if tax_deductible_only:
        query += " AND tax_deductible = 1"

    query += " ORDER BY date DESC, id DESC"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # Ensure clean boolean type for tax_deductible in data_editor
    if not df.empty and "tax_deductible" in df.columns:
        df["tax_deductible"] = df["tax_deductible"].astype(bool)
    return df


def update_receipts_from_df(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Synchronizes updates from an edited DataFrame back into SQLite by matching row id.
    """
    if df.empty or "id" not in df.columns:
        return 0

    updated_count = 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            rec_id = row.get("id")
            if pd.isna(rec_id):
                continue
            cursor.execute(
                """
                UPDATE receipts
                SET vendor = ?,
                    date = ?,
                    total_amount = ?,
                    tax_amount = ?,
                    currency = ?,
                    category = ?,
                    client_project = ?,
                    tax_deductible = ?,
                    line_items_summary = ?,
                    confidence = ?,
                    notes = ?
                WHERE id = ?
                """,
                (
                    str(row.get("vendor", "")).strip(),
                    str(row.get("date", "")).strip(),
                    float(row.get("total_amount", 0.0)),
                    float(row.get("tax_amount", 0.0)),
                    str(row.get("currency", "")).strip().upper()[:3],
                    str(row.get("category", "Other")),
                    str(row.get("client_project", "")).strip(),
                    1 if row.get("tax_deductible") in [True, 1, "1", "true"] else 0,
                    str(row.get("line_items_summary", ""))[:80],
                    str(row.get("confidence", "low")),
                    str(row.get("notes", "")),
                    int(rec_id),
                ),
            )
            updated_count += 1
        conn.commit()
    return updated_count


def delete_receipts(receipt_ids: List[int], db_path: str = DB_PATH) -> int:
    """Deletes receipts by their primary key IDs."""
    if not receipt_ids:
        return 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in receipt_ids)
        cursor.execute(f"DELETE FROM receipts WHERE id IN ({placeholders})", receipt_ids)
        conn.commit()
        return cursor.rowcount


def delete_all_data(db_path: str = DB_PATH) -> bool:
    """Completely wipes all rows in the receipts table."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM receipts;")
        conn.commit()
        conn.isolation_level = None
        conn.execute("VACUUM;")
    finally:
        conn.close()
    return True


def get_distinct_values(column_name: str, db_path: str = DB_PATH) -> List[str]:
    """Retrieves unique non-empty values for a given column (e.g. client_project, currency)."""
    allowed_cols = {"currency", "client_project", "category"}
    if column_name not in allowed_cols:
        return []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT DISTINCT {column_name} FROM receipts WHERE {column_name} IS NOT NULL AND {column_name} != '' ORDER BY {column_name}"
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]
