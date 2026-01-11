import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence


def plot_equity_curves(curve_train: Sequence[float], curve_test: Sequence[float], title: str = "Equity Curves", save_path: Path | None = None):
    plt.figure(figsize=(12, 6))
    plt.plot(curve_train, label="Train (in-sample) equity")
    plt.plot(curve_test, label="Test (out-of-sample) equity")
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_single_curve(curve: Sequence[float], title: str = "Equity Curve", save_path: Path | None = None):
    plt.figure(figsize=(10, 6))
    plt.plot(curve, label=title)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
