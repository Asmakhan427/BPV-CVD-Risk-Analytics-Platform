"""Package installation script for the BPV-CVD Risk Analytics Platform."""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def _read_requirements() -> list:
    req_file = ROOT / "requirements.txt"
    reqs = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="bpv-cvd-analytics",
    version="1.0.0",
    description="Machine learning analytics platform for blood pressure variability and cardiovascular "
                "risk assessment in hemodialysis patients (Montoya et al., 2025).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BPV-CVD Analytics Team",
    packages=find_packages(include=["bpv_cvd", "bpv_cvd.*"]),
    py_modules=["run_analysis", "run_pipeline"],
    python_requires=">=3.9",
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "bpv-cvd-analyze=run_analysis:main",
            "bpv-cvd-pipeline=run_pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
    ],
)
