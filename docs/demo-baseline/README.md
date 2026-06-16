# Demo Baseline

This directory captures the expected JSON output for one verified run of each
implemented phase. Runtime side effects such as SQLite databases, discovery
logs, and concept-library JSON files are written under `artifacts/`, which is
intentionally ignored by git.

Regenerate the baseline from the repository root with:

```powershell
.\scripts\run.ps1 -Steps 5 -Json
.\scripts\run-twin.ps1 -InterventionVariable action.force -InterventionValue 2 -QueryVariable green.position -Json
.\scripts\run-embodied.ps1 -Database artifacts\baseline-phase03.db -Json
.\scripts\run-closedloop.ps1 -Experiments 6 -Database artifacts\baseline-phase04.db -Json
.\scripts\run-discovery.ps1 -Friction 0.25 -Log artifacts\baseline-phase05-discovery.jsonl -Json
.\scripts\run-generalist.ps1 -SourceFriction 0.30 -TargetFriction 0.05 -ConceptLibrary artifacts\baseline-phase06-concept-library.json -Json
.\scripts\run-historian.ps1 -HistoryLength 6 -TrueMass 2.5 -HiddenFriction 0.25 -Json
.\scripts\run-composer.ps1 -Friction 0.25 -Restitution 0.65 -CompoundCount 8 -Json
.\scripts\run-theorist.ps1 -LawObservations 20 -Gravity 9.81 -Json
.\scripts\run-universalist.ps1 -Json
.\scripts\run-strategist.ps1 -HistoryLength 6 -TrueMass 2.5 -HiddenFriction 0.25 -Json
.\scripts\run-cartographer.ps1 -Json
.\scripts\run-narrator.ps1 -Json
.\scripts\run-curator.ps1 -Json
.\scripts\run-collaborator.ps1 -Json
.\scripts\run-integrator.ps1 -Json
```
