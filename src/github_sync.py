
import shutil, subprocess, datetime, base64
from pathlib import Path

REPO = "guptuv/Mars-Terrain-Diffusion"
REPO_URL = "https://github.com/" + REPO + ".git"
NOTEBOOK = Path("/content/drive/MyDrive/Colab Notebooks/Mars_Data_Engine.ipynb")
AUTHOR_NAME, AUTHOR_EMAIL = "Yuvraj Gupta", "guptuv" + chr(64) + "users.noreply.github.com"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply" + chr(64) + "anthropic.com>"
WORK = Path("/content/_github_sync")      # temporary clone (not on Drive)
MAX_MB = 20                                 # skip any single file bigger than this
BT = chr(96) * 3                            # three backticks, for code blocks in the README

PHASES = [("phase0", "Setup: GPU, pinned libraries, Drive layout, seeds, run logging"),
          ("phase1", "Real data: DoMars16k, class mapping, leakage check, CTX-grouped split"),
          ("phase2", "Baseline classifier (real data only)"),
          ("phase3", "LoRA fine-tuning of Stable Diffusion 1.5"),
          ("phase4", "Generation + classifier filtering"),
          ("phase5", "Set-level evaluation: FID, KID, precision/recall, Vendi"),
          ("phase6", "Downstream test: real vs real + synthetic"),
          ("phase7", "Optional extensions")]

def _auth_args(token):
    """Send the token as an HTTP header (not inside the URL), so it never lands in .git/config."""
    basic = base64.b64encode(("x-access-token:" + token).encode()).decode()
    return ["-c", "http.extraHeader=Authorization: Basic " + basic]

def _run(cmd, cwd, secret=""):
    """Run a git command; if it fails, show the error with the secret hidden."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = r.stderr + r.stdout
        if secret: msg = msg.replace(secret, "***")
        raise RuntimeError("git step failed: " + " ".join(c for c in cmd if "Basic" not in c)[:80] + "\n" + msg)
    return r.stdout

def collect(root: Path):
    """Map 'path in repo' -> 'file on Drive' for everything we publish."""
    files = {}
    def add(src, dest):
        if src.exists() and src.is_file() and src.stat().st_size < MAX_MB * 1e6:
            files[dest] = src
    add(root / "code" / "mars_utils.py", "src/mars_utils.py")
    add(root / "code" / "github_sync.py", "src/github_sync.py")
    add(root / "PROJECT_NOTES.md", "notes/PROJECT_NOTES.md")
    for p in sorted((root / "notes").glob("*.md")):        add(p, "notes/" + p.name)
    for p in sorted((root / "run_configs").glob("*.json")): add(p, "configs/" + p.name)
    for p in sorted((root / "data" / "splits").glob("*")):  add(p, "splits/" + p.name)
    for p in sorted((root / "results").glob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".csv", ".json", ".md"}: add(p, "results/" + p.name)
    add(NOTEBOOK, "notebooks/Mars_Data_Engine.ipynb")
    return files

def readme(root: Path):
    done = lambda k: (root / (".done_" + k)).exists()
    rows = "\n".join(f"| {i} | {d} | {'done' if done(k) else 'to do'} |" for i, (k, d) in enumerate(PHASES))
    return f"""# Mars-Terrain-Diffusion: a Mars data engine

Generating **labeled synthetic Martian terrain images** with a LoRA-fine-tuned Stable Diffusion model,
filtering them with a classifier trained on real data, judging them with set-level metrics
(FID, KID, precision/recall, Vendi), and testing whether they actually help a downstream classifier.

This project follows up an earlier prompt-only Stable Diffusion v1.4 study (about 730 images, guidance scale 0,
pairwise metrics only, no downstream test) and fixes its main weaknesses.

## Pipeline status
| Phase | What | Status |
|---|---|---|
{rows}

Full, auto-generated notes and numbers: [notes/PROJECT_NOTES.md](notes/PROJECT_NOTES.md)

## Key decisions so far
- **Data:** DoMars16k from Mars-Bench (Hugging Face Mirali33/mb-domars16k): 16,150 CTX tiles (~6 m/px), 15 classes.
- **Leakage found:** every CTX image in the official test split also has tiles in train. We use a
  **CTX-image-grouped split** (splits/geo_split.csv) as the main split; the official split is kept only for comparison.
- **Classes without real data** (subsurface cavities, layered deposits) are not generated, because they cannot be checked.

## Repository layout
{BT}
notebooks/   Colab notebook (run top to bottom, T4 GPU)
src/         helper code (seeding, run logging, GitHub sync)
notes/       running project notes, one section per phase
configs/     JSON settings of every run (seeds, versions, hyper-parameters)
splits/      train/val/test split files, class mapping, test-set lock (hashes)
results/     tables and figures
{BT}
Datasets, checkpoints, and generated images are kept on Google Drive (too large for git).

## Reproduce
Open notebooks/Mars_Data_Engine.ipynb in Colab with a T4 GPU and run the cells in order.
Library versions are pinned in the install cell and recorded in configs/phase0_environment.json.

## Data credit
DoMars16k: Wilhelm et al., 2020; distributed via Mars-Bench (Purohit et al., 2025), CC-BY-4.0.

_Last synced: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_
"""

GITIGNORE = "data/raw/\ncheckpoints/\ngenerated/\n*.safetensors\n*.ckpt\n*.pt\n*.parquet\n__pycache__/\n.ipynb_checkpoints/\n"

def sync(root, message, token, only_prefixes=None):
    """Copy project files into a fresh clone, commit with message, and push.
    only_prefixes: if given, only repo paths starting with one of these are added in this commit."""
    root = Path(root)
    auth = _auth_args(token)
    secret = auth[1].split("Basic ")[1]
    if WORK.exists(): shutil.rmtree(WORK)
    _run(["git"] + auth + ["clone", "-q", REPO_URL, str(WORK)], "/content", secret)
    _run(["git", "checkout", "-q", "-B", "main"], WORK)
    _run(["git", "config", "user.name", AUTHOR_NAME], WORK)
    _run(["git", "config", "user.email", AUTHOR_EMAIL], WORK)

    files = collect(root)
    if only_prefixes:
        files = {d: s for d, s in files.items() if d.startswith(tuple(only_prefixes))}
    for dest, src in files.items():
        (WORK / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, WORK / dest)
    (WORK / "README.md").write_text(readme(root))
    (WORK / ".gitignore").write_text(GITIGNORE)

    _run(["git", "add", "-A"], WORK)
    if not _run(["git", "status", "--porcelain"], WORK).strip():
        print("Nothing changed - no commit made."); return
    _run(["git", "commit", "-q", "-m", message + TRAILER], WORK)
    _run(["git"] + auth + ["push", "-q", "origin", "main"], WORK, secret)
    print(f"Pushed: {message}  ({len(files)} files)  ->  https://github.com/{REPO}")
