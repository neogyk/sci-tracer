# CLI Tool Reference

## Event generation

| Tool | Purpose |
|---|---|
| `MadGraph5_aMC@NLO` (`mg5_aMC`) | Matrix-element generation; produces LHE files |
| `Pythia8` | Parton shower + hadronisation; reads LHE, writes HepMC |
| `Delphes` | Fast detector simulation; reads HepMC, writes ROOT |

## Analysis

| Tool | Purpose |
|---|---|
| `ROOT` / `PyROOT` | Histogram filling, file I/O (`TFile`, `TTree`, `TH1`) |
| `uproot` + `awkward` | Pure-Python ROOT I/O |
| `coffea` | columnar HEP analysis framework |

## Utilities

| Tool | Purpose |
|---|---|
| `python analysis/<script>.py` | Run your event-selection script |
| `python -m pytest analysis/` | Run any unit tests you write |

## Directory conventions

```
analysis/   — Python event-selection code
data/       — output ROOT files from Delphes
sims/       — generator / shower / detector cards (*.dat, *.cmnd, *.tcl)
results/    — YAML templates to fill with predicted yields
```
