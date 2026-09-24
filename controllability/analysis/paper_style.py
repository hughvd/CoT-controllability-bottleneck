"""Shared appearance and export settings for paper figures."""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

PAPER_FIGURES = Path(__file__).resolve().parents[2] / "paper" / "figures"
SQUARE_SIZE = (5.5, 5.5)
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12
CELL_SIZE = 12
HEATMAP = "RdYlGn"
# Model identity stays fixed across sentence/token prefill and rank figures.
MODEL_COLORS = {
    "qwen14": "#7fb3ff", "qwen32": "#1f4eb0",
    "deepseek": "#2ca02c", "glm": "#9467bd",
    "gptoss": "#ff7f0e", "olmo31": "#d62728",
}
BEHAVIOR_LABELS = {
    "uppercase_thinking": "Uppercase", "leetspeak": "Leetspeak",
    "spanish": "Spanish", "chinese": "Chinese",
    "prefix_sentences": "Sentence prefixes", "no_first_person": "No first person",
    "omit_word": "Omit word", "pirate": "Pirate", "yoda": "Yoda",
    "jar_jar_binks": "Jar Jar Binks",
}


def apply_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": TICK_SIZE,
        "axes.labelsize": LABEL_SIZE, "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE, "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def export_figure(fig, folder, name, *, square=False, paper_name=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    bbox = None if square else "tight"
    png = folder / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches=bbox)
    fig.savefig(folder / f"{name}.pdf", bbox_inches=bbox)
    shutil.copy2(png, PAPER_FIGURES / f"{paper_name or name}.png")
    plt.close(fig)
    print(f"wrote {png}; copied PNG to {PAPER_FIGURES}")
