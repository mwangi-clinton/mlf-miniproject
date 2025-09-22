# Mapping Healthcare Access in Kenya

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Description

This repository contains code and resources for a mini-project developed as part of the **Mathematical Foundations of Machine Learning Workshop** led by Prof. Neil Lawrence.

The project explores **healthcare access in Kenya** by integrating multiple datasets (health facilities, population census, financial and household data) to investigate disparities in resource distribution and accessibility.

Healthcare access in Kenya is structured through a tiered system of facilities, ranging from **level 1 (community-based services)** to **level 6 (national referral hospitals)**. Specialized healthcare is primarily available at **level 4 facilities and above**, which provide advanced services.

This mini-project seeks to **quantify access to specialized emergency care** by combining population census data with health facility records. Using regression models, the study estimates the **probability that individuals in different regions can reach and utilize higher-level facilities**. The analysis provides a **data-driven approach** to understanding healthcare access in Kenya and highlights the role of facility distribution and population characteristics in shaping access.

The notebook (notebook/mini_project.ipynb) is structured into three main sections:

- **Access** – Data acquisition and feature creation
- **Assess** – Data assessment, quality checks, and exploratory analysis
- **Address** – Predictive modeling and answering questions on access to specialized care

---

## ⚙️ Prerequisites

- Python 3.9 or higher
- Poetry for dependency management (optional but recommended)
- Jupyter Notebook or any Python IDE for running the scripts

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mwangi-clinton/mlf-miniproject.git
cd mlf-miniproject

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
# Or, using Poetry
poetry install
poetry shell
```
