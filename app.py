"""
ReceiptSorter - Private, Local Receipt & Invoice Sorter
100% offline, local open-weight vision model inference via Ollama.
No cloud APIs, no external scripts or CDNs, zero data leakage.
"""

import io
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
from PIL import Image
import streamlit as st

import db
import extractor

# Initialize local SQLite database
db.init_db()


# ==========================================
# STEP 1: CUSTOM OFFLINE THEME & CSS
# ==========================================
def load_css() -> None:
    """
    Injects custom CSS adhering strictly to the offline rule:
    - Uses system font stack only (-apple-system, SF Pro, Segoe UI, Roboto)
    - Zero external network requests (no Google Fonts, no CDNs)
    - Rounded cards with subtle borders & soft shadows
    - Solid violet buttons with smooth hover states
    - Dashed violet file uploader and polished tabs
    """
    st.markdown(
        """
        <style>
        /* System font stack and container sizing */
        html, body, [class*="css"] {
            font-family: -apple-system, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
            color: #E8EAF0;
        }

        /* Hide default Streamlit decoration, menu & footer */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        [data-testid="stDecoration"] { display: none; }

        /* Constrain container width with generous padding */
        .main .block-container {
            max-width: 1200px;
            padding-top: 1.75rem;
            padding-bottom: 4rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Hero styling */
        .hero-title {
            font-size: 2.75rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
            color: #FFFFFF;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: #94A3B8;
            margin-bottom: 1.25rem;
        }

        /* Hero pill badges */
        .hero-badges-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            margin-bottom: 2rem;
        }
        .pill-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            background: #1E2330;
            color: #CBD5E1;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .pill-status-ready {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(34, 197, 94, 0.14);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.32);
        }
        .pill-status-error {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.14);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.32);
        }

        /* Rounded Cards */
        .glass-card {
            background-color: #171B26;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
            margin-bottom: 20px;
        }

        /* Confidence Badges */
        .conf-badge-high {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .conf-badge-medium {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(234, 179, 8, 0.15);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.3);
        }
        .conf-badge-low {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .review-label {
            display: inline-block;
            margin-left: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #fb7185;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Clean solid buttons without gradients */
        div.stButton > button:first-child {
            background-color: #7C5CFF !important;
            color: #FFFFFF !important;
            border: 1px solid #7C5CFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.25rem !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #6A44F5 !important;
            border-color: #6A44F5 !important;
        }

        /* File uploader dashed violet border & drop area */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(124, 92, 255, 0.45) !important;
            border-radius: 16px !important;
            padding: 18px !important;
            background: rgba(23, 27, 38, 0.5) !important;
        }

        /* Empty state styling */
        .empty-state-box {
            text-align: center;
            padding: 48px 24px;
            background-color: #171B26;
            border: 1px dashed rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            margin: 20px 0;
        }
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 12px;
        }
        .empty-state-hint {
            color: #94A3B8;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state() -> None:
    """Sets up default session state structures."""
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = extractor.DEFAULT_MODEL_NAME
    if "default_currency" not in st.session_state:
        st.session_state["default_currency"] = "USD"
    if "categories_list" not in st.session_state:
        st.session_state["categories_list"] = list(extractor.DEFAULT_CATEGORIES)
    if "clients_list" not in st.session_state:
        st.session_state["clients_list"] = ["Acme Corp", "Studio Alpha", "Personal", "Internal"]
    if "extraction_cache" not in st.session_state:
        st.session_state["extraction_cache"] = {}
    if "scanned_cards" not in st.session_state:
        st.session_state["scanned_cards"] = []
    if "scanned_images" not in st.session_state:
        st.session_state["scanned_images"] = []
    if "pending_sample_files" not in st.session_state:
        st.session_state["pending_sample_files"] = []
    if "duplicate_warning_queue" not in st.session_state:
        st.session_state["duplicate_warning_queue"] = None


# ==========================================
# STEP 2: HEADER / HERO SECTION
# ==========================================
def render_hero(ollama_running: bool, model_installed: bool, active_model: str) -> None:
    """Renders the top hero with title, subtitle, and badges."""
    st.markdown('<div class="hero-title">ReceiptSorter</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Sort receipts privately. Your data never leaves this laptop.</div>',
        unsafe_allow_html=True,
    )

    # Status pill logic
    if not ollama_running:
        status_html = '<div class="pill-status-error">● Ollama not running</div>'
    elif not model_installed:
        status_html = f'<div class="pill-status-error">● Model missing: {active_model}</div>'
    else:
        status_html = f'<div class="pill-status-ready">● Model ready: {active_model}</div>'

    badges_html = f"""
    <div class="hero-badges-row">
        <div class="pill-badge">🔒 100% offline</div>
        <div class="pill-badge">⚡ Open-source AI</div>
        <div class="pill-badge">💰 Zero cloud cost</div>
        {status_html}
    </div>
    """
    st.markdown(badges_html, unsafe_allow_html=True)


# ==========================================
# STEP 3: TAB 1 - SCAN TAB
# ==========================================
def render_scan_tab(ollama_ready: bool, active_model: str) -> None:
    """Handles receipt file uploading, sample testing, extraction, and card review."""
    st.markdown("### 📤 Upload & Process Receipts")
    st.write(
        "Drop receipt photos or invoices below. The local vision model extracts fields, detects taxes, and caches images."
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload receipts or invoices",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="scan_file_uploader",
            help="Images are processed strictly on your laptop.",
        )
    with col_sample:
        st.write("&nbsp;")
        if st.button("📁 Try with sample receipts", use_container_width=True):
            sample_dir = "sample_receipts"
            loaded_samples = []
            if os.path.exists(sample_dir):
                for f in sorted(os.listdir(sample_dir)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        fpath = os.path.join(sample_dir, f)
                        with open(fpath, "rb") as fh:
                            loaded_samples.append((f, fh.read()))
            if loaded_samples:
                st.session_state["pending_sample_files"] = loaded_samples
                st.toast(f"Loaded {len(loaded_samples)} sample receipts! Click 'Process receipts' below.", icon="📥")
            else:
                st.warning("No sample images found in `sample_receipts/`.")

    # Determine files to process
    files_to_process = []
    if uploaded_files:
        for uf in uploaded_files:
            files_to_process.append((uf.name, uf.getvalue()))
    elif st.session_state["pending_sample_files"]:
        files_to_process = st.session_state["pending_sample_files"]

    if files_to_process:
        st.info(f"Ready to process **{len(files_to_process)}** receipt(s).")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        process_clicked = st.button(
            "⚡ Process receipts",
            type="primary",
            disabled=(len(files_to_process) == 0 or not ollama_ready),
            use_container_width=True,
        )

    # Process batch
    if process_clicked and files_to_process:
        progress_bar = st.progress(0.0)
        status_msg = st.empty()
        extracted_cards = []
        extracted_images = []

        total_files = len(files_to_process)
        batch_start = time.time()

        for idx, (filename, file_bytes) in enumerate(files_to_process):
            status_msg.text(f"Processing {filename} ({idx + 1}/{total_files})...")

            # Prepare image
            try:
                img_bytes, pil_img = extractor.prepare_image(io.BytesIO(file_bytes))
            except Exception as img_err:
                st.warning(f"Could not open image '{filename}': {img_err}. Skipping.")
                continue

            # Model call with retry and session caching
            receipt_dict, was_cached, elapsed = extractor.extract_receipt_with_retry(
                img_bytes,
                filename=filename,
                model_name=active_model,
                cache=st.session_state["extraction_cache"],
                allowed_categories=st.session_state["categories_list"],
            )

            # Assign default currency if missing
            if not receipt_dict.get("currency"):
                receipt_dict["currency"] = st.session_state["default_currency"]

            extracted_cards.append(receipt_dict)
            extracted_images.append(pil_img)
            progress_bar.progress((idx + 1) / total_files)

        total_time = time.time() - batch_start
        status_msg.success(
            f"✅ Processed {len(extracted_cards)} receipt(s) in {total_time:.2f}s!"
        )
        st.session_state["scanned_cards"] = extracted_cards
        st.session_state["scanned_images"] = extracted_images
        st.session_state["pending_sample_files"] = []
        st.toast(f"Extracted {len(extracted_cards)} receipts ready for review!", icon="🎉")

    # Render Scanned Cards
    if st.session_state["scanned_cards"]:
        st.divider()
        st.markdown("### 📋 Review Extracted Receipts")
        st.caption("Review and edit any field before saving to your permanent ledger.")

        cards = st.session_state["scanned_cards"]
        imgs = st.session_state["scanned_images"]

        # Card container loop
        for i, card in enumerate(cards):
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            conf = (card.get("confidence") or "low").lower()
            if conf == "high":
                conf_badge = '<span class="conf-badge-high">High Confidence</span>'
            elif conf == "medium":
                conf_badge = '<span class="conf-badge-medium">Medium Confidence</span>'
            else:
                conf_badge = '<span class="conf-badge-low">Low Confidence</span><span class="review-label">⚠️ Needs Review</span>'

            col_img, col_fields = st.columns([1, 2])

            with col_img:
                if i < len(imgs) and imgs[i] is not None:
                    st.image(imgs[i], use_container_width=True, caption=card.get("file_name", f"Receipt {i+1}"))
                else:
                    st.write("📷 Preview not available")
                st.markdown(f"**Confidence:** {conf_badge}", unsafe_allow_html=True)

            with col_fields:
                card["vendor"] = st.text_input(
                    "Vendor Name",
                    value=card.get("vendor", ""),
                    key=f"card_vendor_{i}",
                )
                card["date"] = st.text_input(
                    "Date (YYYY-MM-DD)",
                    value=card.get("date", ""),
                    key=f"card_date_{i}",
                )

                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    card["total_amount"] = st.number_input(
                        "Total Amount",
                        value=float(card.get("total_amount", 0.0)),
                        min_value=0.0,
                        step=0.01,
                        key=f"card_total_{i}",
                    )
                with sub_c2:
                    card["tax_amount"] = st.number_input(
                        "Tax Amount",
                        value=float(card.get("tax_amount", 0.0)),
                        min_value=0.0,
                        step=0.01,
                        key=f"card_tax_{i}",
                    )

                sub_c3, sub_c4 = st.columns(2)
                with sub_c3:
                    card["currency"] = st.text_input(
                        "Currency (e.g. USD, EUR, INR)",
                        value=card.get("currency", st.session_state["default_currency"]),
                        key=f"card_curr_{i}",
                    ).upper()[:3]
                with sub_c4:
                    cat_list = st.session_state["categories_list"]
                    cat_index = 0
                    if card.get("category") in cat_list:
                        cat_index = cat_list.index(card["category"])
                    card["category"] = st.selectbox(
                        "Expense Category",
                        options=cat_list,
                        index=cat_index,
                        key=f"card_cat_{i}",
                    )

                sub_c5, sub_c6 = st.columns([2, 1])
                with sub_c5:
                    client_opts = [""] + st.session_state["clients_list"]
                    current_client = card.get("client_project", "")
                    cl_idx = client_opts.index(current_client) if current_client in client_opts else 0
                    card["client_project"] = st.selectbox(
                        "Client / Project",
                        options=client_opts,
                        index=cl_idx,
                        key=f"card_client_{i}",
                    )
                with sub_c6:
                    st.write("&nbsp;")
                    card["tax_deductible"] = st.checkbox(
                        "Tax Deductible",
                        value=bool(card.get("tax_deductible", False)),
                        key=f"card_taxded_{i}",
                    )

                card["line_items_summary"] = st.text_input(
                    "Line Items Summary (max 80 chars)",
                    value=card.get("line_items_summary", ""),
                    key=f"card_items_{i}",
                )[:80]

                card["notes"] = st.text_input(
                    "Notes",
                    value=card.get("notes", ""),
                    key=f"card_notes_{i}",
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Duplicate detection check & save logic
        st.divider()

        # Handle pending duplicate warning if user clicked save previously
        if st.session_state["duplicate_warning_queue"]:
            dup_info = st.session_state["duplicate_warning_queue"]
            st.warning(
                f"⚠️ **Duplicate Warning:** Found {len(dup_info['duplicates'])} receipt(s) with identical Vendor, Date, and Amount in your Ledger:\n"
                + "\n".join([f"- **{d['vendor']}** on {d['date']} for {d['total_amount']} {d.get('currency','')}" for d in dup_info["duplicates"]])
            )
            col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
            with col_d1:
                if st.button("Save All Anyway", type="primary"):
                    db.save_receipts_batch(st.session_state["scanned_cards"])
                    saved_n = len(st.session_state["scanned_cards"])
                    st.session_state["scanned_cards"] = []
                    st.session_state["scanned_images"] = []
                    st.session_state["duplicate_warning_queue"] = None
                    st.toast(f"Saved {saved_n} receipts to Ledger!", icon="✅")
                    st.rerun()
            with col_d2:
                if st.button("Skip Duplicates & Save Remaining"):
                    to_save = [c for c in st.session_state["scanned_cards"] if c not in dup_info["duplicates"]]
                    if to_save:
                        db.save_receipts_batch(to_save)
                        st.toast(f"Saved {len(to_save)} new receipts (skipped duplicates)!", icon="✅")
                    else:
                        st.info("All scanned receipts were duplicates. Nothing to save.")
                    st.session_state["scanned_cards"] = []
                    st.session_state["scanned_images"] = []
                    st.session_state["duplicate_warning_queue"] = None
                    st.rerun()
            with col_d3:
                if st.button("Cancel"):
                    st.session_state["duplicate_warning_queue"] = None
                    st.rerun()
        else:
            col_save, col_clear = st.columns([2, 1])
            with col_save:
                if st.button("💾 Save all to ledger", type="primary", use_container_width=True):
                    # Duplicate check
                    duplicates = []
                    for card in st.session_state["scanned_cards"]:
                        existing = db.check_duplicate(
                            card.get("vendor", ""),
                            card.get("date", ""),
                            card.get("total_amount", 0.0),
                        )
                        if existing:
                            duplicates.append(card)

                    if duplicates:
                        st.session_state["duplicate_warning_queue"] = {"duplicates": duplicates}
                        st.rerun()
                    else:
                        count = db.save_receipts_batch(st.session_state["scanned_cards"])
                        st.session_state["scanned_cards"] = []
                        st.session_state["scanned_images"] = []
                        st.toast(f"Saved {count} receipts to Ledger!", icon="✅")
                        st.rerun()

            with col_clear:
                if st.button("Clear Scanned Cards", use_container_width=True):
                    st.session_state["scanned_cards"] = []
                    st.session_state["scanned_images"] = []
                    st.rerun()

    elif not files_to_process:
        # Friendly empty state
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">🧾</div>
                <h4>No receipts scanned yet</h4>
                <div class="empty-state-hint">
                    Drag and drop receipt photos above, or click <strong>'Try with sample receipts'</strong> to see instant local extraction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# STEP 3: TAB 2 - LEDGER TAB
# ==========================================
def render_ledger_tab() -> None:
    """Displays the persistent SQLite ledger with filters, metric cards, and editable table."""
    st.markdown("### 📚 Receipts Ledger")
    st.caption("Permanent local database stored in `data/receipts.db`. Edits and new entries persist across restarts.")

    # Fetch all records
    df = db.get_all_receipts()

    if df.empty:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">📂</div>
                <h4>Your ledger is currently empty</h4>
                <div class="empty-state-hint">
                    Head over to the <strong>Scan</strong> tab to upload and save receipts to your permanent ledger.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Metric cards row
    total_receipts = len(df)
    avg_receipt = df["total_amount"].mean() if total_receipts > 0 else 0.0

    # Group spending by currency to never add mixed currencies together
    currency_totals = df.groupby("currency")["total_amount"].sum().to_dict()
    curr_summary_str = ", ".join([f"{amt:,.2f} {curr}" for curr, amt in currency_totals.items() if curr])
    if not curr_summary_str:
        curr_summary_str = f"{df['total_amount'].sum():,.2f}"

    tax_deductible_total = df[df["tax_deductible"] == True]["total_amount"].sum()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Spent", curr_summary_str)
    with m2:
        st.metric("Receipts Count", total_receipts)
    with m3:
        st.metric("Average Receipt", f"{avg_receipt:,.2f}")
    with m4:
        st.metric("Tax Deductible Total", f"{tax_deductible_total:,.2f}")

    st.divider()

    # Search and Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        search_kw = st.text_input("🔍 Search", placeholder="Vendor, notes, client...", key="ledger_search")
    with f_col2:
        cats = ["All"] + db.get_distinct_values("category")
        selected_cat = st.selectbox("Category", options=cats, key="ledger_filter_cat")
    with f_col3:
        clients = ["All"] + db.get_distinct_values("client_project")
        selected_client = st.selectbox("Client / Project", options=clients, key="ledger_filter_client")
    with f_col4:
        currencies = ["All"] + db.get_distinct_values("currency")
        selected_curr = st.selectbox("Currency", options=currencies, key="ledger_filter_curr")

    # Filtered dataframe
    filtered_df = db.get_all_receipts(
        search=search_kw,
        category=selected_cat,
        client_project=selected_client,
        currency=selected_curr,
    )

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")

    # Table columns configuration
    cols_order = [
        "id",
        "file_name",
        "vendor",
        "date",
        "total_amount",
        "tax_amount",
        "currency",
        "category",
        "client_project",
        "tax_deductible",
        "line_items_summary",
        "confidence",
        "notes",
    ]
    display_df = filtered_df[[c for c in cols_order if c in filtered_df.columns]].copy()

    edited_ledger = st.data_editor(
        display_df,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "file_name": st.column_config.TextColumn("File", disabled=True),
            "vendor": st.column_config.TextColumn("Vendor", required=True),
            "date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
            "total_amount": st.column_config.NumberColumn("Total", format="%.2f", min_value=0.0, step=0.01),
            "tax_amount": st.column_config.NumberColumn("Tax", format="%.2f", min_value=0.0, step=0.01),
            "currency": st.column_config.TextColumn("Currency"),
            "category": st.column_config.SelectboxColumn("Category", options=st.session_state["categories_list"]),
            "client_project": st.column_config.TextColumn("Client / Project"),
            "tax_deductible": st.column_config.CheckboxColumn("Tax Deductible"),
            "line_items_summary": st.column_config.TextColumn("Line Items (max 80 chars)"),
            "confidence": st.column_config.SelectboxColumn("Confidence", options=extractor.ALLOWED_CONFIDENCE),
            "notes": st.column_config.TextColumn("Notes"),
        },
        use_container_width=True,
        num_rows="dynamic",
        key="ledger_data_editor",
    )

    btn_c1, btn_c2, _ = st.columns([2, 2, 4])
    with btn_c1:
        if st.button("💾 Save Edits to Database", key="btn_save_ledger_edits"):
            updated_n = db.update_receipts_from_df(edited_ledger)
            st.toast(f"Updated {updated_n} receipt(s) in SQLite database!", icon="✅")
            st.rerun()

    with btn_c2:
        # Multi-row deletion tool
        selected_ids = st.multiselect(
            "Select IDs to Delete",
            options=filtered_df["id"].tolist(),
            placeholder="Select receipt IDs...",
            key="delete_ids_select",
        )
        if selected_ids and st.button(f"🗑️ Delete {len(selected_ids)} Selected", type="secondary"):
            db.delete_receipts(selected_ids)
            st.toast(f"Deleted {len(selected_ids)} receipts from database.", icon="🗑️")
            st.rerun()


# ==========================================
# STEP 3: TAB 3 - REPORTS TAB
# ==========================================
def render_reports_tab() -> None:
    """Renders month selectors, Altair charts, top vendors, and Excel/CSV exports."""
    st.markdown("### 📊 Financial Reports & Analytics")
    st.caption("Visual breakdowns and formatted accounting exports generated locally.")

    df = db.get_all_receipts()
    if df.empty:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">📊</div>
                <h4>No receipts available to generate reports</h4>
                <div class="empty-state-hint">Scan and save receipts in the Scan tab first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Extract available months from date field (YYYY-MM)
    df["month"] = df["date"].astype(str).str.extract(r"^(\d{4}-\d{2})")[0].fillna("Unspecified")
    all_months = ["All Months"] + sorted([m for m in df["month"].unique() if m != "Unspecified"], reverse=True)
    if "Unspecified" in df["month"].values:
        all_months.append("Unspecified")

    selected_month = st.selectbox("📅 Select Reporting Month", options=all_months, key="reports_month_select")

    rep_df = df if selected_month == "All Months" else df[df["month"] == selected_month]

    if rep_df.empty:
        st.info(f"No receipts found for {selected_month}.")
        return

    st.divider()

    # CRITICAL: Totals per currency shown SEPARATELY (never add INR and USD together)
    st.markdown("#### 💵 Spending Totals by Currency")
    curr_breakdown = rep_df.groupby("currency", as_index=False).agg(
        Total_Spent=("total_amount", "sum"),
        Tax_Total=("tax_amount", "sum"),
        Transactions=("id", "count"),
    )
    cols = st.columns(max(len(curr_breakdown), 1))
    for idx, row in curr_breakdown.iterrows():
        c_name = row["currency"] or "Unspecified Currency"
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding:16px;">
                    <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase;">{c_name}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#A78BFA; margin:6px 0;">{row['Total_Spent']:,.2f}</div>
                    <div style="font-size:0.8rem; color:#64748B;">Tax: {row['Tax_Total']:,.2f} ({row['Transactions']} receipts)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Visualizations using Altair with dark violet/blue palette
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📌 Spending by Category")
        cat_spending = rep_df.groupby("category", as_index=False)["total_amount"].sum()
        cat_chart = (
            alt.Chart(cat_spending)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("category:N", sort="-y", title="Category"),
                y=alt.Y("total_amount:Q", title="Total Spent"),
                color=alt.Color(
                    "category:N",
                    scale=alt.Scale(
                        range=["#7C5CFF", "#3B82F6", "#60A5FA", "#818CF8", "#A78BFA", "#C084FC", "#93C5FD"]
                    ),
                    legend=None,
                ),
                tooltip=["category", "total_amount"],
            )
            .properties(height=320)
            .configure_axis(
                labelColor="#E8EAF0",
                titleColor="#E8EAF0",
                gridColor="rgba(255, 255, 255, 0.08)",
                domainColor="rgba(255, 255, 255, 0.15)",
            )
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(cat_chart, use_container_width=True)

    with chart_col2:
        st.markdown("##### 📈 Spending Over Time")
        time_spending = (
            rep_df[rep_df["date"] != ""]
            .groupby("date", as_index=False)["total_amount"]
            .sum()
            .sort_values("date")
        )
        if not time_spending.empty:
            time_chart = (
                alt.Chart(time_spending)
                .mark_line(color="#7C5CFF", point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("total_amount:Q", title="Total Spent"),
                    tooltip=["date", "total_amount"],
                )
                .properties(height=320)
                .configure_axis(
                    labelColor="#E8EAF0",
                    titleColor="#E8EAF0",
                    gridColor="rgba(255, 255, 255, 0.08)",
                    domainColor="rgba(255, 255, 255, 0.15)",
                )
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(time_chart, use_container_width=True)
        else:
            st.info("No dated receipts to plot spending over time.")

    # Top 5 Vendors Table
    st.markdown("##### 🏢 Top 5 Vendors")
    top_vendors = (
        rep_df.groupby("vendor", as_index=False)
        .agg(
            Total_Spent=("total_amount", "sum"),
            Receipt_Count=("id", "count"),
            Currency=("currency", lambda s: ", ".join(sorted(set(s)))),
        )
        .sort_values(by="Total_Spent", ascending=False)
        .head(5)
    )
    top_vendors["Total_Spent"] = top_vendors["Total_Spent"].apply(lambda v: f"{v:,.2f}")
    st.dataframe(top_vendors, use_container_width=True, hide_index=True)

    st.divider()

    # Export Buttons: CSV, Excel (.xlsx with openpyxl), and Tax-Deductible Export
    st.markdown("### 💾 Export Reports")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # 1. Download CSV
    with col_exp1:
        csv_bytes = rep_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name=f"receipts_{selected_month}_{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 2. Download Excel with openpyxl (All Receipts sheet + Summary sheet)
    with col_exp2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            rep_df.to_excel(writer, sheet_name="All Receipts", index=False)
            cat_summary = (
                rep_df.groupby("category", as_index=False)["total_amount"]
                .sum()
                .rename(columns={"total_amount": "Total Spent"})
            )
            curr_summary = (
                rep_df.groupby("currency", as_index=False)["total_amount"]
                .sum()
                .rename(columns={"total_amount": "Total Spent"})
            )
            cat_summary.to_excel(writer, sheet_name="Summary by Category", index=False)
            curr_summary.to_excel(writer, sheet_name="Summary by Currency", index=False)

        st.download_button(
            label="📊 Download Excel (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"receipts_report_{selected_month}_{timestamp_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # 3. Tax-Deductible-Only Export
    with col_exp3:
        tax_df = rep_df[rep_df["tax_deductible"] == True]
        tax_csv = tax_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"🧾 Tax-Deductible Only ({len(tax_df)})",
            data=tax_csv,
            file_name=f"tax_deductible_{selected_month}_{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================
# STEP 3: TAB 4 - SETTINGS TAB
# ==========================================
def render_settings_tab(active_model: str) -> None:
    """Configuration for local vision model, default currency, lists, and database wipe."""
    st.markdown("### ⚙️ Settings & Configuration")

    st.markdown("#### 🤖 Vision Model")
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        new_model = st.text_input(
            "Ollama Vision Model",
            value=active_model,
            help="Any vision model installed in Ollama (e.g. gemma3:4b, qwen2.5vl:7b, llama3.2-vision:11b)",
            key="settings_model_input",
        ).strip()
        if new_model != active_model:
            st.session_state["model_name"] = new_model

    with col_m2:
        st.write("&nbsp;")
        if st.button("🧪 Test Model", use_container_width=True):
            with st.spinner("Pinging model via Ollama..."):
                ok, latency_ms, msg = extractor.test_model_connection(new_model)
                if ok:
                    st.toast(f"Model test success! Latency: {latency_ms:.0f} ms", icon="✅")
                    st.success(f"**Success!** {msg}")
                else:
                    st.error(f"**Connection Failed:** {msg}")

    st.divider()

    # Default Currency & Lists
    st.markdown("#### 💼 Defaults & Organization")
    col_cur, col_cats = st.columns(2)
    with col_cur:
        st.session_state["default_currency"] = st.text_input(
            "Default Currency Code",
            value=st.session_state["default_currency"],
            max_chars=3,
        ).upper()

    with col_cats:
        cats_str = st.text_input(
            "Expense Categories (comma-separated)",
            value=", ".join(st.session_state["categories_list"]),
        )
        if cats_str:
            st.session_state["categories_list"] = [c.strip() for c in cats_str.split(",") if c.strip()]

    clients_str = st.text_input(
        "Clients & Projects (comma-separated)",
        value=", ".join(st.session_state["clients_list"]),
    )
    if clients_str:
        st.session_state["clients_list"] = [c.strip() for c in clients_str.split(",") if c.strip()]

    st.divider()

    # Danger Zone: Delete all data
    st.markdown("#### ⚠️ Danger Zone")
    st.caption("Permanently wipes all saved receipts from the SQLite database. Cannot be undone.")
    confirm_wipe = st.checkbox("I understand this will permanently delete all records in the ledger database.")
    if st.button("🗑️ Delete All Data", type="secondary", disabled=not confirm_wipe):
        db.delete_all_data()
        st.session_state["scanned_cards"] = []
        st.session_state["scanned_images"] = []
        st.session_state["extraction_cache"] = {}
        st.toast("All ledger data has been permanently cleared.", icon="🗑️")
        st.rerun()

    st.divider()

    # About Box
    st.markdown("#### ℹ️ About ReceiptSorter")
    st.markdown(
        f"""
        <div class="glass-card">
            <h4>Private, Zero-Cloud Document AI</h4>
            <p>
                ReceiptSorter runs <strong>100% locally on your machine</strong> using the <code>{active_model}</code>
                open-weight multimodal model served via <strong>Ollama</strong>.
            </p>
            <ul>
                <li><strong>No Cloud APIs:</strong> Your invoices, tax IDs, and confidential vendor data never traverse the internet.</li>
                <li><strong>Offline Resilient:</strong> Works completely without an internet connection.</li>
                <li><strong>Zero API Costs:</strong> No monthly subscriptions or metered per-token charges.</li>
                <li><strong>Model Swapability:</strong> Easily switch between open weights like <code>gemma3:4b</code> or <code>qwen2.5vl:7b</code>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN APPLICATION ROUTING
# ==========================================
def main():
    st.set_page_config(
        page_title="ReceiptSorter",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    load_css()

    active_model = st.session_state["model_name"]
    ollama_running, model_installed, _ = extractor.check_ollama_status(active_model)
    ollama_ready = ollama_running and model_installed

    # Render Hero Section
    render_hero(ollama_running, model_installed, active_model)

    # Render Tabs
    tab_scan, tab_ledger, tab_reports, tab_settings = st.tabs(
        ["Scan", "Ledger", "Reports", "Settings"]
    )

    with tab_scan:
        render_scan_tab(ollama_ready, active_model)

    with tab_ledger:
        render_ledger_tab()

    with tab_reports:
        render_reports_tab()

    with tab_settings:
        render_settings_tab(active_model)


if __name__ == "__main__":
    main()
