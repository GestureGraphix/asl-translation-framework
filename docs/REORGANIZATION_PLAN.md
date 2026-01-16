# File Reorganization Plan

## Current Issues
- Many markdown files in root directory (15+ files)
- Status files, guides, and documentation mixed together
- Hard to find relevant documentation

## Proposed Structure

```
asl-translation-framework/
├── README.md                    # Main project README (stays in root)
├── requirements.txt             # Dependencies (stays in root)
├── setup.py                     # Setup script (stays in root)
│
├── docs/
│   ├── paper/                   # Paper-related files
│   │   ├── ASL Modeling.tex     # Main paper (moved from root)
│   │   └── research_summary.tex # Already here
│   │
│   ├── guides/                  # How-to guides and tutorials
│   │   ├── PHASE1_GUIDE.md
│   │   ├── QUICK_START_COLAB.md
│   │   ├── MANUAL_FIXES_STEP_BY_STEP.md
│   │   └── NOTEBOOK_FIXES_SUMMARY.md
│   │
│   ├── status/                  # Status and progress tracking
│   │   ├── STATUS.md
│   │   ├── ABLATION_STUDY_STATUS.md
│   │   ├── TRAINING_ANALYSIS.md
│   │   └── TRAINING_DIAGNOSIS.md
│   │
│   ├── planning/                # Planning and analysis documents
│   │   ├── NEXT_STEPS.md
│   │   ├── NEXT_STEPS_ANALYSIS.md
│   │   ├── PAPER_GOALS_ROADMAP.md
│   │   └── PAPER_ALIGNMENT_CHECK.md
│   │
│   ├── reference/               # Reference documentation
│   │   ├── CLAUDE.md            # Main implementation guide
│   │   ├── implementation_notes.md (moved from root)
│   │   └── architecture.md      # Already here
│   │
│   └── ... (keep existing PDFs, etc.)
│
├── scripts/
│   ├── diagnostic_code.py       # Moved from root
│   └── ... (existing scripts)
│
└── ... (rest of structure unchanged)
```

## Files to Move

### From root to docs/paper/
- `ASL Modeling.tex` → `docs/paper/ASL Modeling.tex`

### From root to docs/guides/
- `PHASE1_GUIDE.md` → `docs/guides/PHASE1_GUIDE.md`
- `QUICK_START_COLAB.md` → `docs/guides/QUICK_START_COLAB.md`
- `MANUAL_FIXES_STEP_BY_STEP.md` → `docs/guides/MANUAL_FIXES_STEP_BY_STEP.md`
- `NOTEBOOK_FIXES_SUMMARY.md` → `docs/guides/NOTEBOOK_FIXES_SUMMARY.md`

### From root to docs/status/
- `STATUS.md` → `docs/status/STATUS.md`
- `ABLATION_STUDY_STATUS.md` → `docs/status/ABLATION_STUDY_STATUS.md`
- `TRAINING_ANALYSIS.md` → `docs/status/TRAINING_ANALYSIS.md`
- `TRAINING_DIAGNOSIS.md` → `docs/status/TRAINING_DIAGNOSIS.md`

### From root to docs/planning/
- `NEXT_STEPS.md` → `docs/planning/NEXT_STEPS.md`
- `NEXT_STEPS_ANALYSIS.md` → `docs/planning/NEXT_STEPS_ANALYSIS.md`
- `PAPER_GOALS_ROADMAP.md` → `docs/planning/PAPER_GOALS_ROADMAP.md`
- `PAPER_ALIGNMENT_CHECK.md` → `docs/planning/PAPER_ALIGNMENT_CHECK.md`

### From root to docs/reference/
- `CLAUDE.md` → `docs/reference/CLAUDE.md`
- `implementation_notes.md` → `docs/reference/implementation_notes.md`

### From root to scripts/
- `diagnostic_code.py` → `scripts/diagnostic_code.py`

## Files to Keep in Root
- `README.md` - Main entry point
- `.gitignore` - Git config
- `requirements.txt` - Dependencies
- `setup.py` - Setup script

## After Reorganization

Update references:
1. Update README.md to point to new locations
2. Update any cross-references in documentation
3. Update notebook paths if they reference docs
