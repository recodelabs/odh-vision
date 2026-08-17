"""
config.py  —  Central configuration for the ODH scan-extraction pipeline.

Every other script imports from here. Edit paths once, everything follows.

Folder layout expected on disk:
    PROJECT_ROOT/
        ODHFILESCANS_master_extraction.xlsx   ← master workbook
        ODHFILESCANS_master_extraction.xlsx.bak
        ODHFILESCANS_audit_log.csv
        ODHFILESCANS_optimized/               ← optimized PDFs
            <Region>/
                <Center>/
                    <filename>.pdf
        _output/                              ← rendered page images (auto-created)
"""

import os, subprocess, sys

# ─── Auto-install dependencies ────────────────────────────────────────────────
def _ensure_deps():
    """Install missing Python packages automatically on first run."""
    needed = []
    try:
        import openpyxl
    except ImportError:
        needed.append("openpyxl>=3.1.0")
    try:
        from PIL import Image
    except ImportError:
        needed.append("Pillow>=10.0.0")
    if needed:
        print(f"Installing missing packages: {', '.join(needed)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + needed)
        print("Done.\n")

    # Check for poppler-utils (pdftoppm) — needed by step 1
    if not _which("pdftoppm"):
        print("NOTE: poppler-utils not found (needed for PDF rendering).")
        print("      Install with:  sudo apt install poppler-utils")
        print("      (or brew install poppler on Mac)\n")

def _which(cmd):
    """Return True if *cmd* is on PATH."""
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

_ensure_deps()

# ─── Base paths ───────────────────────────────────────────────────────────────
# Override with env var  ODH_PROJECT_ROOT  if you keep files elsewhere.
PROJECT_ROOT = os.environ.get(
    "ODH_PROJECT_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)

MASTER_XLSX   = os.path.join(PROJECT_ROOT, "ODHFILESCANS_master_extraction.xlsx")
MASTER_BAK    = MASTER_XLSX + ".bak"
MASTER_LOCK   = MASTER_XLSX + ".lock"          # advisory lock file
AUDIT_LOG     = os.path.join(PROJECT_ROOT, "ODHFILESCANS_audit_log.csv")
OPTIMIZED_DIR = os.path.join(PROJECT_ROOT, "ODHFILESCANS_optimized")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENTS_DIR = os.path.join(PROJECT_ROOT, "_segments")
os.makedirs(SEGMENTS_DIR, exist_ok=True)

EXTRACTIONS_DIR = os.path.join(PROJECT_ROOT, "_extractions")
os.makedirs(EXTRACTIONS_DIR, exist_ok=True)

RECONCILED_DIR = os.path.join(PROJECT_ROOT, "_reconciled")
os.makedirs(RECONCILED_DIR, exist_ok=True)


# ─── Master-spreadsheet column headers (in column order) ─────────────────────
HEADERS = [
    "Record No.",          # A  (0)
    "Region",              # B  (1)
    "Source PDF",          # C  (2)
    "Page",                # D  (3)
    "Center",              # E  (4)
    "Date",                # F  (5)
    "Time in",             # G  (6)
    "Patient name",        # H  (7)
    "Village",             # I  (8)
    "1st time at ODH",     # J  (9)
    "Last care code",      # K  (10)
    "Travel time code",    # L  (11)
    "1st voucher use",     # M  (12)
    "Age (yrs)",           # N  (13)
    "Sex",                 # O  (14)
    "Tests",               # P  (15)
    "Malaria result",      # Q  (16)
    "Sev. malaria",        # R  (17)
    "Diagnosis",           # S  (18)
    "Treatment",           # T  (19)
    "Tab no.",             # U  (20)
    "Cost (UGX)",          # V  (21)
    "Overall confidence",  # W  (22)
    "Notes / uncertainties",  # X  (23)
    "Optimization used",      # Y  (24)
]

HEADER_INDEX = {h: i for i, h in enumerate(HEADERS)}

# Fields that matter most for auditing
AUDIT_FIELDS = [
    "Patient name", "Village", "Age (yrs)", "Sex",
    "Diagnosis", "Malaria result", "Cost (UGX)",
]

# ─── Cell-color flags (ARGB hex from openpyxl fills) ─────────────────────────
#  green  = verified          yellow = uncertain [?]
#  blue   = faded [faded]     orange = very uncertain [??]
#  red    = illegible
FLAG_COLORS = {
    "FFFFFF00",   # yellow
    "FFFFC000",   # orange
    "FFFF0000",   # red
    "FF4472C4",   # blue (faded)
}

VERIFIED_COLOR = "FF00B050"   # green — confirmed correct


# ─── Reserved sheet names ─────────────────────────────────────────────────────
RESERVED_SHEETS = {"Legend", "Progress"}


# ─── Render settings ─────────────────────────────────────────────────────────
RENDER_DPI  = 300
TOTAL_PDFS  = 312             # total source PDFs in the project


# ─── Confidence legend (color name → marker → description) ───────────────────
CONFIDENCE_PALETTE = [
    ("Green",  "[verified]",  "Re-read at full resolution and confirmed — high confidence."),
    ("Yellow", "[?]",         "One token uncertain; best-read, verify before use."),
    ("Blue",   "[faded]",     "Legible but faint / low-contrast scan; read is plausible."),
    ("Orange", "[??]",        "Low confidence / very messy strokes; ambiguous reading."),
    ("Red",    "[illegible]", "Could not be read; cell left blank — not guessed."),
]
