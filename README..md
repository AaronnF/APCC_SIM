# APCC Congestion Control Simulator

This repository contains a Python-based simulator developed for evaluating the Adaptive Predictive Congestion Control (APCC) framework proposed in the accompanying academic report.

The simulator compares APCC with TCP Reno and DCTCP-style congestion control under a shared bottleneck network model.

---

## Requirements

- Python 3.8 or later
- numpy
- matplotlib

---

## Installation

It is recommended to use a Python virtual environment.

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```
---

## Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Usage

Run the simulator using:

```bash
python sim_cc.py
```

The script will execute the simulation and generate performance plots.

## Output

The simulator produces time-series plots for:

- Queueing delay versus time

- Aggregate goodput versus time

- Average congestion window versus time

These outputs are used for experimental evaluation in the accompanying report.

--- 

## Reproducibility

All experiments in the report were generated using this codebase and the configuration parameters provided in the source file. Users may modify simulation parameters to explore alternative network conditions.

--- 

## Author
Aaron Fernandez

---

## License

This repository is provided for academic evaluation purposes.