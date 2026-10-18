# ReceiptSorter 🧾

> **Sort receipts and invoices privately on your own laptop using local open-weight vision models. 100% offline, zero cloud calls, zero API costs.**

ReceiptSorter is a private, offline-first expense management and receipt extraction application designed for **freelancers, independent contractors, and small business owners**. Upload receipt photos or invoice scans, and a local open-weight vision model extracts the vendor, date, total amount, tax amount, currency, expense category, and line items summary into an interactive card review interface, a persistent SQLite ledger, visual analytics, and multi-format reports.

Everything runs on your device. **No cloud APIs, no API keys, no internet connection required.**

---

## 📸 Screenshots

<!-- Place screenshots or screen recordings here -->
```
+-----------------------------------------------------------------------------------+
|  ReceiptSorter                                                                    |
|  Sort receipts privately. Your data never leaves this laptop.                     |
|  [🔒 100% offline] [⚡ Open-source AI] [💰 Zero cloud cost] [● Model ready: gemma3:4b]|
+-----------------------------------------------------------------------------------+
|  [ Scan ]   [ Ledger ]   [ Reports ]   [ Settings ]                               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   +--------------------------+    +------------------------------------------+    |
|   |                          |    | Vendor: Blue Bottle Coffee               |    |
|   |  [ Receipt Image ]       |    | Date:   2026-10-14                       |    |
|   |                          |    | Total:  18.17 USD      Tax: 1.42 USD     |    |
|   |  Confidence: High (98%)  |    | Cat:    Meals          Client: Studio A  |    |
|   +--------------------------+    +------------------------------------------+    |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

- **100% Offline by Design**: Strict system font stack, inline styling, and zero runtime web dependencies. Works identically with Wi-Fi switched off.
- **Card-Based Visual Review**: Two-column layout pairing the actual receipt image alongside editable extracted fields.
- **Enhanced Vision Prompt with Auto-Retry**: Automatically captures `tax_amount` and `line_items_summary`. If initial model output fails strict JSON parsing, an immediate retry prompt is triggered before fallback.
- **Image Session Caching**: Computes SHA-256 hashes of receipt images so re-scans in the same session complete instantaneously without re-running model inference.
- **Persistent SQLite Ledger (`data/receipts.db`)**: Saves records locally across application restarts. Includes search, multi-field filters, inline editing, and deletion.
- **Intelligent Duplicate Detection**: Flags receipts sharing identical vendor, date, and amount before writing to the database, giving the user the option to skip or save anyway.
- **Altair Visual Analytics**: Categorical expense breakdowns and temporal spending charts styled with a dark violet/blue palette.
- **Multi-Currency Safety**: Automatically calculates separate subtotals per currency code so international expenses (e.g. INR vs. USD vs. EUR) are never erroneously summed together.
- **Accounting Exports**: One-click downloads for raw CSV, tax-deductible-only CSV, and multi-sheet formatted Excel spreadsheets (`.xlsx`) via `openpyxl`.

---

## Why Open-Source AI and Local Inference Matter

Handling financial documents requires strict confidentiality. Financial transactions, vendor names, client identifiers, tax details, and personal expense habits should never be exposed unnecessarily to cloud providers or third-party APIs:

1. **Absolute Data Privacy**: Client financial data, invoices, and sensitive receipts stay strictly on your device. Nothing is uploaded to remote servers or used to train third-party models.
2. **Zero Per-Receipt Costs**: No metered API charges or subscription tiers. Process 5 receipts or 5,000 receipts without paying a cent.
3. **Works Anywhere, Completely Offline**: File your taxes, audit books, or organize expenses on a flight, in a cafe with spotty Wi-Fi, or during network outages.
4. **Model Freedom & Swapability**: You aren't locked into one vendor's proprietary API or pricing model. Easily swap the underlying vision model to newer or larger open weights (e.g., `gemma3:4b`, `qwen2.5vl:7b`, `llama3.2-vision:11b`) with a single setting.

---

## Requirements

- **macOS** (Apple Silicon M1/M2/M3/M4 recommended for fast GPU-accelerated local inference) or Linux/Windows
- **Python 3.9+**
- **Ollama**: Free, local model runner ([ollama.com](https://ollama.com))
- Recommended model: `gemma3:4b` (vision-capable open-weight model)

---

## Setup (Step-by-Step for macOS)

### 1. Install Ollama
Download and install Ollama from [ollama.com/download](https://ollama.com/download). Launch the Ollama application once to ensure the background service is running.

### 2. Pull the Vision Model
Open Terminal and pull Google's vision-capable `gemma3:4b` model:
```bash
ollama pull gemma3:4b
```
*(Optional: You can also pull other vision models such as `ollama pull qwen2.5vl:7b`.)*

### 3. Set Up the Python Virtual Environment
Navigate to this folder in your terminal:
```bash
cd /path/to/receipt-sorter
```
Create a virtual environment (if not already created):
```bash
python3 -m venv .venv
```
Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

### 5. Launch ReceiptSorter
Run the Streamlit application:
```bash
streamlit run app.py
```
Streamlit will automatically open the app in your default browser at `http://localhost:8501`.

---

## App Walkthrough & Navigation

### 1. Scan Tab
- Drag and drop your receipt photos or invoice scans (`.png`, `.jpg`, `.jpeg`, `.webp`).
- Or click **"Try with sample receipts"** to immediately test bundled sample invoices.
- Click **"Process receipts"** to run the local vision model with live progress and per-receipt timing.
- Review the extracted cards in the two-column view, make any adjustments, and click **"Save all to ledger"**.

### 2. Ledger Tab
- Access all previously stored receipts from the local SQLite database (`data/receipts.db`).
- Check top-level financial metrics: Total Spent, Receipt Count, Average Receipt, and Tax Deductible Total.
- Filter by category, client/project, currency, or search by keywords.
- Edit values directly in the interactive table and click **"Save Edits to Database"** or delete records by ID.

### 3. Reports Tab
- Select a specific month or view aggregate metrics across all time.
- Review currency-segregated totals, category bar charts, and spending timelines.
- Export your financial data to **CSV**, multi-sheet **Excel (.xlsx)** with summary tabs, or a **Tax-Deductible-Only** report.

### 4. Settings Tab
- Change the active Ollama model name and run the **"Test Model"** latency benchmark.
- Set default currency codes, custom expense categories, and client project tags.
- Use the **"Delete All Data"** button with confirmation safety to reset the local database.

---

## How to Switch Models

You can switch models in two ways:

1. **Directly in the Web UI (Settings Tab)**:
   - Navigate to the **Settings** tab.
   - Enter another installed vision model (e.g. `qwen2.5vl:7b` or `llama3.2-vision:11b`).
   - Click **"Test Model"** to verify connection and response latency.

2. **In Code (`extractor.py`)**:
   - Open `extractor.py` and modify the default configuration variable:
     ```python
     DEFAULT_MODEL_NAME = "gemma3:4b"  # Change to "qwen2.5vl:7b", etc.
     ```

---

## Known Limitations

- **Blurry or Crumpled Receipts**: Poor lighting, extreme motion blur, or heavily creased paper can degrade OCR accuracy.
- **Handwritten Receipts**: Open-weight vision models perform best on printed text; handwritten receipts or signatures may yield lower confidence.
- **Small Model Nuances**: Compact models (e.g. 4B parameters) can occasionally misread similar digits (such as `8` vs `3` or decimal points) or miss faint itemized taxes.
- **Human-in-the-Loop Always Recommended**: Always review the highlighted cards and check low-confidence receipts before filing tax or financial reports.

---

## License

This project is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 YOUR NAME HERE.
