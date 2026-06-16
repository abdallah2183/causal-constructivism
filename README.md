# Causal Constructivism v0.16

A runnable causal-graph-first active inference prototype.

## Implementation Status Dashboard

| Category | Phase Range | Status | Implementation Depth |
| --- | --- | --- | --- |
| **Grounded Physical Core** | Phases 1 - 11 | `Executable Research Core` | Fully implemented, mathematically verified against rigid-body simulators. |
| **Metacognitive Facades** | Phases 12 - 16 | `Research Facade` | Structural stubs / Template-based simulations verifying data flow. |
| **Planned Extensions** | Future Roadmap | `Planned / Future Work` | Purely conceptual specifications of open-ended reasoning. |

---

### Grounded Physical Core (Phases 1 - 11)
These phases are fully implemented using physical simulation backends (MuJoCo, 1D rigid-body engines), symbolic mathematical engines, and Bayesian active inference networks. All parameters, prediction errors, and state revisions are computed dynamically from actual sensor observations.

*   **Phase 1**: Active scalar mass inference.
*   **Phase 2**: Adds the Twin World: multi-object collision physics, particle uncertainty, graph surgery, and counterfactual queries.
*   **Phase 3**: Adds the Embodied Vision foundation: persistence, synthetic RGB-D proposals, object permanence, temporal graph chains, and an optional MuJoCo adapter.
*   **Phase 4**: Closes the loop with embodied active inference: MuJoCo-rendered observations drive tracking, tracking drives EFE action selection, selected 3D forces execute in MuJoCo, and action outcomes update grounded mass beliefs.
*   **Phase 5**: Adds the first Discoverer slice: neural structure proposals over experiment-log anomalies, with conservative scoring and grounded integration of a discovered friction concept.
*   **Phase 6**: Adds the first Generalist slice: learned concepts can be serialized, transferred into a materially different environment by signature, contradicted by new evidence, and revised through graph versioning.
*   **Phase 7**: Adds the first Historian slice: experiment histories can be stitched into causal graphs, replayed under policy interventions with shared exogenous trials, and compared as counterfactual scientific methodologies.
*   **Phase 8**: Adds the first Composer slice: compound observations are scored against single-concept and pairwise physical models, allowing the system to discover friction and restitution together and integrate a grounded interaction relation.
*   **Phase 9**: Adds the first Theorist slice: symbolic expression templates are fitted and scored against observations, allowing the system to discover a grounded pendulum square-root law instead of only fitting parameters to known concepts.
*   **Phase 10**: Adds the first Universalist slice: structurally similar discovered laws can be abstracted into a grounded cross-domain principle that predicts a new domain before direct observation.
*   **Phase 11**: Adds the first Strategist slice: discovery policies can be generated, counterfactually evaluated over replayed experiment histories, and adopted only when they improve discovery efficiency while preserving grounding.

---

### Ontological & Metacognitive Facades (Phases 12 - 16)
These phases outline the system's metacognitive architecture. To maintain a standard-library-only style without external LLM or heavy NLP dependencies, these modules are implemented as structural stubs. They simulate narratives, research questions, and agent debates using graph-based templates and rule-based queries.

*   **Phase 12 (Cartographer)**: Building a navigable meta-ontology graph over discovered concepts and relations, allowing the system to query conceptual similarity, trace cross-domain analogies, and score distinguishing experiments.
*   **Phase 13 (Narrator)**: Generating grounded natural language explanations from the Cartographer's meta-ontology graph, where every claim is traceable to observations, laws, or inferences through provenance paths using text-formatting templates.
*   **Phase 14 (Curator)**: Autonomous research agenda formation by reading the Narrator's explanations, identifying gaps in the Cartographer's conceptual map via node-degree thresholds, and prioritizing open questions.
*   **Phase 15 (Collaborator)**: Multi-agent scientific discourse where competing internal scientist agents debate hypotheses, weigh evidence, and resolve disagreements through simplified parameter convergence.
*   **Phase 16 (Integrator)**: An end-to-end cognitive orchestration loop that coordinates all systems into a single unified agent lifecycle backed by simulator rollouts.

The implementation deliberately starts with a narrow 1D physical experiment so the graph, inference, planning, learning, and grounding contracts can be tested without hiding unresolved behavior behind a large perception stack.

## Implemented

- Typed causal property graph with node and edge validation.
- Immutable history for superseded node versions.
- Scalar Gaussian beliefs and linear-Gaussian pairwise factors.
- Residual-scheduled, bidirectional Gaussian belief propagation.
- Approximate variational free-energy reporting.
- Penalized linear structure proposals from observation history.
- Expected-free-energy action selection with epistemic and pragmatic terms.
- 1D rigid-body simulator and Bayesian active mass inference.
- Grounding traces from inferred nodes to observations or axioms.
- Deterministic CLI demonstration and standard-library test suite.
- Vector Gaussian state marginals produced by particle moment matching.
- Multi-object 1D rigid-body dynamics with impulse collisions.
- Actual/counterfactual twin graph construction with soft interventions.
- Shared exogenous sampling across actual and counterfactual rollouts.
- Position, velocity, and collision-probability counterfactual queries.
- Bayesian contact-relation proposals integrated into the causal graph.
- Transactional SQLite graph snapshots with historical-node restoration.
- Synthetic RGB-D projection with configurable depth and pixel uncertainty.
- Physics-predicted object tracking with optimal global association.
- Birth, occlusion, reappearance, and conservative identity-loss handling.
- Temporal `evolves_to` graph chains with grounded observed and hidden states.
- Optional MuJoCo 3D force, contact, state, and RGB-D rendering adapter.
- Reversible MuJoCo state snapshots for side-effect-free candidate planning.
- Pruned 3D push primitive library scored by expected free energy.
- Closed-loop embodied mass inference over three visually tracked objects.
- Action and plan provenance linked into the grounding graph.
- JSONL experiment logs for discovery training and replay.
- Free-energy anomaly detection over persistent prediction residuals.
- Small trainable neural proposer constrained by a graph-operation grammar.
- Conservative model-evidence scoring before structure integration.
- Grounded discovery of a latent `friction` property and law.
- Serializable concept library with revision history.
- Signature-based transfer of a learned concept across environments.
- Evidence-gated causal revision with immutable graph supersession.
- Multi-environment friction benchmark with pre/post revision error reporting.
- Experiment-level graph stitching with explicit policy and library snapshots.
- Deterministic history replay with shared exogenous trials.
- History-level policy counterfactuals over friction-before-mass discovery.
- Meta-scientific metrics for mass-error, prediction-error, grounding, and
  experiments saved.
- Bounded multi-concept physical grammar including friction, restitution,
  damping, spring constant, and center of mass entries.
- Hierarchical model comparison between single-concept and pairwise compound
  explanations.
- Grounded discovery of friction plus restitution from slide-collision
  observations.
- Grounded interaction relation for friction-loss modulation of restitution.
- Symbolic expression tree evaluation over variables, constants, powers, and
  square roots.
- Bounded symbolic equation search with least-squares parameter fitting.
- Model comparison over constant, linear, polynomial, and square-root law
  templates.
- Grounded discovery of `T = 2*pi*sqrt(L/g)` from pendulum observations.
- Expression-tree pattern abstraction across grounded law instances.
- Cross-domain harmonic-motion unification over pendulum and spring laws.
- Prediction and validation of the physically correct LC oscillator period
  `T = 2*pi*sqrt(L*C)`.
- Meta-law grounding through grounded law instances and confirmed predictions.
- Discovery-policy generation by bounded perturbation of action force,
  anomaly threshold, and discovery horizon.
- Counterfactual policy evaluation with the Historian replay substrate.
- Meta-policy adoption gated by discovery-efficiency improvement and composite
  methodology score.
- Grounded strategy-decision nodes with policy-score provenance.

## Run

Use Python 3.11 or newer:

```powershell
.\scripts\run.ps1 -Steps 5
```

JSON output:

```powershell
.\scripts\run.ps1 -Steps 5 -Json
```

Run the Twin World benchmark:

```powershell
.\scripts\run-twin.ps1 -InterventionVariable action.force `
    -InterventionValue 2 -QueryVariable green.position -Json
```

Query whether a specific collision would occur:

```powershell
.\scripts\run-twin.ps1 -InterventionValue 2 `
    -QueryVariable collision.red.green -Json
```

Run the embodied perception and permanence benchmark:

```powershell
.\scripts\run-embodied.ps1 -Database artifacts\embodied-vision.db -Json
```

Install the optional real MuJoCo backend:

```powershell
python -m pip install -e ".[embodied]"
```

Run the native starter scene:

```powershell
.\scripts\run-mujoco.ps1 -Json
```

Run the closed-loop embodied scientist benchmark:

```powershell
.\scripts\run-closedloop.ps1 -Experiments 6 -Json
```

Run the Phase 5 discovery benchmark:

```powershell
.\scripts\run-discovery.ps1 -Friction 0.25 -Json
```

Run the Phase 6 generalist transfer and revision benchmark:

```powershell
.\scripts\run-generalist.ps1 -SourceFriction 0.30 -TargetFriction 0.05 `
    -ConceptLibrary artifacts\generalist.json -Json
```

Run the Phase 7 historian counterfactual benchmark:

```powershell
.\scripts\run-historian.ps1 -HistoryLength 6 -TrueMass 2.5 `
    -HiddenFriction 0.25 -Json
```

Run the Phase 8 composer multi-concept benchmark:

```powershell
.\scripts\run-composer.ps1 -Friction 0.25 -Restitution 0.65 `
    -CompoundCount 8 -Json
```

Run the Phase 9 theorist symbolic law benchmark:

```powershell
.\scripts\run-theorist.ps1 -LawObservations 20 -Gravity 9.81 -Json
```

Run the Phase 10 universalist cross-domain unification benchmark:

```powershell
.\scripts\run-universalist.ps1 -Json
```

Run the Phase 11 strategist self-improvement benchmark:

```powershell
.\scripts\run-strategist.ps1 -HistoryLength 6 -TrueMass 2.5 `
    -HiddenFriction 0.25 -Json
```

Run the Phase 12 cartographer conceptual-atlas benchmark:

```powershell
.\scripts\run-cartographer.ps1 -Json
```

Run the Phase 13 narrator explanation benchmark:

```powershell
.\scripts\run-narrator.ps1 -Json
```

Run the Phase 14 curator research-agenda benchmark:

```powershell
.\scripts\run-curator.ps1 -Json
```

Run the Phase 15 collaborator discourse benchmark:

```powershell
.\scripts\run-collaborator.ps1 -Json
```

Run the Phase 16 end-to-end integrator benchmark:

```powershell
.\scripts\run-integrator.ps1 -Json
```

Install as a local package when a working Python installation is available:

```powershell
python -m pip install -e .
causal-constructivism --steps 5
```

## Test

The suite has no third-party runtime requirement:

```powershell
.\scripts\test.ps1
```

The PowerShell scripts prefer Codex's bundled Python runtime when present and
otherwise use the `python` command on `PATH`.

## Vertical Slice

Each experiment step:

1. Scores candidate push actions by expected posterior entropy plus a
   displacement penalty.
2. Executes the selected force in the simulator.
3. Updates a grid posterior over mass using `a = F / m`.
4. Creates an immutable observation node and links it to the mass property.
5. Runs event-driven Gaussian message passing.
6. Audits the inferred mass for a path to observational or axiomatic roots.

The CLI reports the selected force, measured acceleration, posterior mass,
posterior uncertainty, free energy, and grounding status.

## Package Layout

| Module | Responsibility |
| --- | --- |
| `models.py` | Beliefs, typed nodes/edges, and result contracts |
| `graph.py` | Graph mutation, adjacency, versioning, grounding paths |
| `inference.py` | Gaussian belief propagation and free energy |
| `learning.py` | Structure proposal, scoring, and integration |
| `planning.py` | Expected-free-energy action scoring |
| `physics.py` | 1D world and Bayesian mass estimator |
| `twin_physics.py` | Multi-object dynamics and particle collision factor |
| `counterfactual.py` | Twin graph, abduction, surgery, and prediction |
| `contact_learning.py` | Pairwise contact-relation model selection |
| `twin_system.py` | Phase 2 integration facade |
| `persistence.py` | Transactional SQLite graph snapshots |
| `embodied.py` | 3D environment, action, camera, and proposal contracts |
| `perception.py` | Synthetic RGB-D uncertainty projection |
| `tracking.py` | Object permanence and optimal assignment |
| `temporal.py` | Persistent identities and temporal state graph chains |
| `mujoco_adapter.py` | Optional official MuJoCo Python backend |
| `embodied_system.py` | Phase 3 synthetic integration benchmark |
| `agency.py` | Phase 4 closed-loop EFE planning and active mass inference |
| `discovery.py` | Phase 5 anomaly detection, neural proposal, and concept integration |
| `generalist.py` | Phase 6 concept library, transfer, contradiction, and revision |
| `historian.py` | Phase 7 history replay, policy twinning, and meta-science metrics |
| `composer.py` | Phase 8 multi-concept model comparison and interaction discovery |
| `theorist.py` | Phase 9 symbolic expression search and law grounding |
| `universalist.py` | Phase 10 cross-domain law unification and meta-law grounding |
| `strategist.py` | Phase 11 policy generation, replay evaluation, and adoption |
| `cartographer.py` | Phase 12 meta-ontology graph neighborhood and analogies (Facade) |
| `narrator.py` | Phase 13 template-based explanation generation from ontology (Facade) |
| `curator.py` | Phase 14 topological gap detection and research agendas (Facade) |
| `collaborator.py` | Phase 15 multi-agent convergence debate orchestration (Facade) |
| `integrator.py` | Phase 16 end-to-end cognitive orchestration loop (Orchestrator) |
| `audit.py` | Confidence and grounding classification |
| `system.py` | End-to-end sensorimotor experiment loop |
| `cli.py` | Runnable demonstration |

## Scope Boundaries

This is not yet the complete supersystem described in the source
specification. The following require separate implementation and validation:

- RGB-D segmentation and object tracking.
- General-purpose nonlinear or discrete variational factors beyond collisions.
- Symbolic `do`-calculus simplification and arbitrary graph query planning.
- Persistent hot/cold graph storage.
- Parallel inference workers and read-copy-update concurrency.
- ROS2 or physical hardware integration.
- Unrestricted contact-rich manipulation, grasping, tipping, and rolling.
- Learned dynamics approximations for fast large-candidate planning.
- Counterfactual queries over arbitrary full closed-loop MuJoCo experiment
  histories beyond the current synthetic friction/mass history.
- Open-ended neural structure proposal beyond the current bounded concept
  grammars.
- Open-ended symbolic law invention beyond the current bounded expression
  templates.
- Open-ended cross-domain unification beyond the current harmonic-motion
  pattern.
- Transfer and revision for multiple interacting concepts beyond the current
  friction-only Generalist.
- Claims of zero hallucination or transformer-level energy comparisons.

The current grounding audit provides traceability; it does not prove that an
axiom, sensor, model, or inference result is factually correct.

## Twin World Semantics

Counterfactual execution follows abduction, action, prediction:

1. Condition uncertain initial states and physical parameters on evidence.
2. Run the actual particle world.
3. Clone the causal graph.
4. Remove incoming edges to each intervened variable and fix its belief.
5. Re-run the world with identical particle seeds and changed mechanisms.
6. Moment-match output particles and audit the query node's provenance.

Supported interventions include action force/duration and object mass, radius,
restitution, friction, position, or velocity. Supported queries are object
position, object velocity, any collision, or a named pair collision.

## Embodied Vision Foundation

The Phase 3 benchmark renders synthetic object segments with depth, projects
them into uncertain 3D proposals, tracks two cubes through a deliberate
occlusion, writes observed and predicted state nodes into a temporal graph, and
persists the graph to SQLite. It validates identity continuity and perceptual
grounding before introducing renderer and contact-solver variability.

The MuJoCo adapter follows the official native Python API and has been verified
against MuJoCo 3.9.0: models load through
`MjModel.from_xml_*`, state lives in `MjData`, stepping uses `mj_step`, and
offscreen RGB/depth/segmentation uses `Renderer`. See the
[MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html).

## Closed Loop

The Phase 4 benchmark uses a controlled three-object MuJoCo scene with slide
joints. This keeps mass identifiable while still exercising the full embodied
loop:

1. render RGB-D and segmentation from MuJoCo,
2. project segments into uncertain object proposals,
3. track persistent identities,
4. score pruned 3D push candidates with MuJoCo forward rollouts,
5. execute the selected force,
6. update Bayesian mass posteriors from observed acceleration,
7. write plan/action/outcome provenance into the causal graph,
8. persist the final graph snapshot.

Current verified result: all three object masses are inferred within 15% after
six pushes, with confident mass audits and zero ungrounded nodes after
restoration.

## Discoverer

The Phase 5 slice introduces a neural proposal contract without pretending to
solve open-ended science. It targets one measurable discovery: when a constant
velocity model repeatedly fails because sliding objects stop, the system
detects the anomalous subgraph, trains a small neural classifier on synthetic
friction/no-friction curricula, proposes a `friction` property, scores the
proposal by model-evidence improvement and neural confidence, then integrates a
friction law and property only if the score clears a conservative threshold.

Current verified result: eight sliding records with hidden friction `0.25`
produce a grounded `block_001.friction` property with coefficient `0.25`, high
neural confidence, strong evidence gain, and zero ungrounded nodes. Constant
velocity records produce no proposal.

## Generalist

The Phase 6 slice tests whether a learned concept survives contact with a new
environment without being treated as immutable truth. The benchmark first
discovers `friction` on a source surface, stores it in a concept library, then
transfers it into a target surface when the dynamical signature indicates sliding
deceleration. The transfer starts broad and speculative because the material changed.

New target observations are then scored against the transferred coefficient. If the prediction error provides enough evidence gain, the system supersedes the transferred friction node, installs the revised coefficient, records the revision in the concept library, and audits the new node through target observations and the friction-law axiom.

Current verified result: transferring `friction=0.30` from wood into ice starts
with prediction error `0.21465`, revises the coefficient to `0.05`, reduces
post-revision error to `0.0`, records evidence gain above `100`, and leaves the
revised friction node confidently grounded.

## Historian

The Phase 7 slice asks a meta-scientific question: would the system have
inferred mass more efficiently if it had discovered friction before using mass
estimates? The benchmark builds a deterministic sequence of shared exogenous
trials, replays an actual policy that discovers friction after enough anomalous
sliding records, then replays a counterfactual policy where the friction concept
is available before mass inference.

Each replay writes experiment nodes, policy nodes, concept-library snapshots,
mass-estimate nodes, and `evolves_to` links into a causal graph. Actual and
counterfactual histories use the same trial IDs, object mass, hidden friction,
durations, and initial velocities. Only the policy differs.

Current verified result: with six experiments, true mass `2.5`, and hidden
friction `0.25`, the actual policy discovers friction at experiment `4` and has
mean mass error `0.12069`. The counterfactual policy knows friction at
experiment `0`, reduces mean mass error to `0.0`, reduces velocity-prediction
error from `0.14125` to `0.0`, saves four experiments to friction, and leaves
both history graphs fully grounded.

## Composer

The Phase 8 slice tests whether the system can choose between competing
physical explanations when an observation contains more than one phenomenon.
The benchmark observes a block sliding into a wall. Friction reduces the impact
velocity before contact, and restitution controls the rebound velocity after
contact.

The selector fits three explanations: friction only, restitution only, and
friction plus restitution. Each model is scored by evidence gain minus a
complexity penalty. If the pairwise model wins, the graph receives grounded
concept nodes for `friction` and `restitution`, plus a grounded relation node
representing `friction_loss_modulates_restitution`.

Current verified result: eight compound observations with hidden friction
`0.25` and restitution `0.65` select the pairwise model, recover both
parameters exactly in the noiseless benchmark, score `109.74`, beat the
single-concept alternatives, and leave zero ungrounded nodes.

## Theorist

The Phase 9 slice tests whether the system can discover a mathematical law form
from observations. The benchmark observes pendulum periods at different lengths
under known gravity. It fits bounded symbolic templates, including constant,
linear, quadratic, cubic-polynomial, and square-root forms, then scores each
law by evidence gain minus expression complexity plus a small consistency
bonus.

The winning law is integrated as a non-axiomatic law node grounded through the
symbolic-regression observations and a theoretical-consistency node. This keeps
the law revisable while still auditable.

Current verified result: twenty pendulum observations with gravity `9.81`
select `T = 6.28319*sqrt(L / g)`, recover the coefficient `2*pi`, produce
zero RMSE in the noiseless benchmark, score `290.49`, beat the overfit cubic
alternative, and leave zero ungrounded nodes.

## Universalist

The Phase 10 slice tests whether the system can see a shared pattern across
discovered laws. The benchmark supplies two harmonic law instances:
pendulum period `T = 2*pi*sqrt(L/g)` and spring period `T = 2*pi*sqrt(m/k)`.
The pattern abstractor recognizes both as `coefficient*sqrt(inertia/restoring)`
and proposes the meta-law `T = 2*pi*sqrt(inertia/restoring)`.

The meta-law then predicts the LC oscillator period in a new domain. The
implementation uses the physically correct formula `T = 2*pi*sqrt(L*C)`, where
inductance supplies the inertia-like term and inverse capacitance supplies the
restoring term. The meta-law is grounded through the pendulum and spring
instances plus the confirmed LC prediction; if the prediction is left untested,
the audit remains speculative.

Current verified result: two grounded harmonic instances unify into
`harmonic_motion` with confidence `0.90`, predict and confirm
`T = 2*pi*sqrt(L*C)`, audit the meta-law as confident with grounding confidence
`0.85`, and leave zero ungrounded nodes.

## Strategist

The Phase 11 slice tests whether the system can improve its own scientific
methodology in a bounded policy space. Starting from the baseline Historian
policy, the Strategist generates candidate policies that adjust anomaly
sensitivity, action force, and discovery horizon. Each candidate is replayed on
the same shared exogenous history and scored for experiments-to-discovery, mean
mass error, and grounding quality.

A policy is adopted only when it improves the composite methodology score and
clears the configured discovery-efficiency threshold. The decision is written
as a grounded policy node with incoming evidence from replay observations and
policy-score nodes.

Current verified result: the baseline policy discovers friction at experiment
`4` with mean mass error `0.12069`. The selected `baseline.fast_discovery`
policy discovers friction at experiment `3`, reduces mean mass error to
`0.05691`, preserves grounding quality at `1.0`, clears the `1.333x`
efficiency gain threshold, and leaves zero ungrounded nodes.

## Cartographer (Ontology Facade)

The Phase 12 slice introduces the Cartographer: a meta-scientific ontology mapping engine. Instead of treating concepts as isolated graph structures or flat entries in a library, the Cartographer constructs a navigable map of the conceptual space. Nodes in this map represent learned physical properties (mass, friction, restitution, spring constant) and edges represent relations (modulates, predicts, is-analogous-to, contradicts).

> [!NOTE]
> *Implementation Depth*: This module is a facade that navigates a small metadata graph using BFS neighborhood traversal. It does not perform semantic ontology alignment.

The benchmark tests conceptual navigation and experiment query selection. Given a set of competing hypotheses (e.g., whether a velocity drop is caused by restitution or damping), the Cartographer queries its map to identify the topological neighborhood, locate the analog domains, and propose a disambiguating experiment using the Strategist's policy generator.

Proposed verified benchmark: Given an ambiguous sliding-collision trajectory, the Cartographer traverses the ontology, identifies that `friction` and `restitution` are both active, retrieves the composition law from Phase 8, and structures a query that distinguishes them from a pure `damping` model, reducing conceptual ambiguity to zero while maintaining a fully grounded ontology path.

## Narrator (Cognitive Facade)

The Phase 13 slice introduces the Narrator: a natural language scientific communication engine. It translates the Cartographer's conceptual map into grounded, human-readable explanations. Every sentence is explicitly linked to supporting nodes in the causal graph to maintain complete auditable provenance.

> [!NOTE]
> *Implementation Depth*: This module does not use a natural language model (LLM). It instantiates formatting string templates from a predefined template grammar.

The benchmark tests explanation generation and validation. Given the Cartographer's ontology after discovering interacting friction and restitution, the Narrator generates a structured explanation of the physics involved, validating that all claims are backed by observation or axiom nodes.

Proposed verified benchmark: Given the Composer's slide-collision model, the Narrator generates a 5-sentence grounded text explaining how sliding deceleration and collision bounce interact. The explanation audit returns `confident` with grounding quality `1.0` and no ungrounded claims.

## Curator (Cognitive Facade)

The Phase 14 slice introduces the Curator: an autonomous research agenda formation engine. It reads the Narrator's explanations, identifies gaps in the Cartographer's conceptual map, and prioritizes open questions based on expected information gain, connectivity, and feasibility.

> [!NOTE]
> *Implementation Depth*: This module identifies gaps via simple topological node-degree thresholds in the metadata graph.

The benchmark tests gap detection and research agenda proposal. Given the Narrator's explanation of the friction-restitution interaction, the Curator detects an unexplained neighboring damping concept, ranks the priority of investigating it, queries the Cartographer for a distinguishing experiment, and asks the Strategist for a feasible policy.

Proposed verified benchmark: Given the friction-restitution explanation, the Curator identifies `damping` as an unexplored sparse neighborhood gap, proposes the question "Does damping modulate restitution independently of friction?" with priority above threshold, and outputs a grounded `ResearchAgenda` with all items successfully audited.

## Collaborator (Cognitive Facade)

The Phase 15 slice introduces the Collaborator: a multi-agent scientific discourse engine. Causal graphs and ontology structures are shared among internal ScientistAgents with varying priors. The Cartographer's contradicts edges identify disagreements, and the Curator and Strategist resolve empirical disputes.

> [!NOTE]
> *Implementation Depth*: This module coordinates a single-round debate convergence using simplified parameter updates rather than dynamic multi-agent argumentation logic.

The benchmark tests convergence and multi-agent debate resolution. Two agents with competing priors (one favoring friction, one favoring damping) debate a slide-collision velocity drop. They propose a distinguishing experiment, execute it, reconcile their beliefs, and log either consensus or dissent.

Proposed verified benchmark: Given two agents debating friction vs. damping, they propose and execute a compliance experiment. The agent favoring damping converges to the true friction model, adding a `consensus` meta-meta-edge to the graph. The debate audit returns `confident` with zero ungrounded meta-meta-edges.

## Integrator (Orchestrator)

The Phase 16 slice introduces the Integrator: an end-to-end cognitive orchestration loop. It coordinates active inference action selection, perception, concept learning, generalization, policy adaptation, gap detection, and multi-agent debate resolution. Rather than relying on static or mocked convergence updates, it wires real physical simulator rollouts directly into the debate-resolution pipeline to update agent beliefs and record consensus empirically.

> [!NOTE]
> *Implementation Depth*: The Integrator is a control loop linking the facade modules. It demonstrates execution-path connectivity and verifies that physical outcomes (from MuJoCo or the 1D simulator) can successfully resolve multi-agent disputes.

The benchmark tests the entire integrated lifecycle. Given two debating agents, the Integrator executes active simulator rollouts to collect empirical data, updating both agents' posteriors based on physical outcomes. This leads to exact convergence, recording a grounded `consensus` edge in the ontology with a `confident` status and a grounding quality of `1.0`.

## Documentation Principles

When updating this repository's documentation, developers must adhere to these rules:
1.  **Maturity Parity**: Do not treat cognitive facades (Phases 12–16) as having equal operational depth as the physical core (Phases 1–11). Explicitly state the facade nature in any module description.
2.  **Terminology Bounding**: Avoid using production terms. Refer to the system as a "Verified Research Prototype." Do not describe stubs using terms like "Proposed verified benchmark"; instead, use "Illustrative facade validation scenario."
3.  **Grounding vs. Truth**: The grounding audit verifies traceability (provenance) of a node back to observations or axioms. It does not prove that an axiom, sensor, or inference result is factually correct.
