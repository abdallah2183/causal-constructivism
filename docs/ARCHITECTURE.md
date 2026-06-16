# Architecture Decision Record

## Objective

Build the smallest system that exercises the proposed architecture as a
coherent loop:

`observe -> update graph -> infer -> audit -> choose action -> observe`

The first environment is a one-dimensional physics playground. This validates
the core contracts before introducing computer vision, MuJoCo, or distributed
storage.

## Mathematical Kernel

Each scalar node carries a Gaussian prior and posterior:

`q(x) = Normal(mean, variance)`

Supported causal factors are linear Gaussian:

`y = weight * x + bias + noise`

where `noise` is zero-mean Gaussian. Messages are produced in both directions.
A sender computes a cavity belief that excludes the receiver's previous
message, preventing immediate double counting. Incoming Gaussian messages are
fused in natural-parameter form.

The current free-energy value is an operational diagnostic:

`sum(KL(posterior || prior)) + observation negative log likelihood`

It is not a full generalized-coordinate active-inference objective.

## Action Model

The planner minimizes:

`expected posterior entropy + displacement_weight * goal_error^2`

The entropy term is epistemic: stronger pushes generally provide more mass
information. The displacement term is pragmatic: experiments should avoid
unbounded movement. The selected action is executed once, then replanning
occurs from the updated posterior.

## Structure Learning

The initial learner fits a relation:

`target = weight * source + bias`

It compares residual variance against a constant baseline and subtracts a
logarithmic complexity penalty. Only positive-score proposals may be
integrated. This gives structure mutation a concrete model-selection gate
without claiming general ontology invention.

## Grounding

A node is grounded when a directed upstream path reaches:

- an observation node with explicit evidence, or
- a node marked as an axiom.

Audits return the roots, traversed edge IDs, confidence, and one of:
`confident`, `speculative`, or `ungrounded`.

Grounding is provenance, not truth. Sensor calibration, axiom validity, and
model misspecification remain independent evaluation problems.

## Extension Order

1. Add vector-valued Gaussian factors and nonlinear factor adapters.
2. Add SQLite persistence with explicit graph transactions.
3. Add intervention and counterfactual query semantics.
4. Replace the 1D world with a MuJoCo adapter behind the same action and
   observation contracts.
5. Add RGB-D perception as uncertain observation proposals.
6. Profile sparse scheduling before introducing parallel workers.

## Phase 2: Twin World

The first nonlinear adapter is an impulse collision factor. It samples uncertain
mass and initial state beliefs, executes deterministic rigid-body rollouts, and
returns diagonal vector-Gaussian state marginals through moment matching.
Collision occurrence remains a Bernoulli probability instead of being collapsed
into a continuous state estimate.

Counterfactual queries use an explicit twin:

1. Clone the actual graph and condition it on supplied evidence.
2. Clone the conditioned graph for the counterfactual world.
3. Remove incoming edges to intervened variables.
4. Fix their beliefs to narrow Gaussians while preserving outgoing edges.
5. Execute actual and counterfactual particle rollouts with the same random
   seed, thereby sharing sampled exogenous state.
6. Attach the counterfactual result to intervention, dynamics, and physical
   parameter nodes and run the standard grounding audit.

The current particle adapter is deliberately factor-specific. It does not claim
to be a general particle belief-propagation engine, and moment matching loses
state multimodality. Collision probability is retained separately to avoid
erasing the key collided/did-not-collide branch.

Contact relations are learned from repeated pair collision observations. A
Beta-smoothed existence probability and penalized likelihood-ratio score gate
integration. The learned relation is represented as a typed relation node with
incoming `touches` edges from both objects.

## Phase 3: Embodied Vision Foundation

Long-lived temporal graphs are persisted as append-only SQLite snapshots. A
single immediate transaction contains every node, edge, and historical node
version. Foreign keys prevent partial graph commits, WAL mode permits concurrent
readers, and loading reconstructs original IDs and timestamps.

Synthetic perception consumes rendered segmentation regions with depth. Each
region is projected through camera intrinsics into a diagonal 3D Gaussian. The
uncertainty includes depth noise, pixel noise, and within-segment dispersion.

Object permanence uses:

1. constant-velocity prediction,
2. class-gated Mahalanobis association,
3. global minimum-cost assignment,
4. conservative birth and loss thresholds,
5. prediction-only states during occlusion.

Every persistent object owns a time-indexed chain of state nodes. Visible states
receive observation edges from RGB-D proposal nodes; occluded states remain
grounded through the previous state and persistent object identity. Reappearing
detections continue the same chain.

The MuJoCo adapter is optional because the native package and OpenGL backend are
environment-specific. Its contract is concrete: apply a 3D force and induced
torque through `xfrc_applied`, step with `mj_step`, read named-body state,
extract contact forces with `mj_contactForce`, and render RGB, depth, and
segmentation through `Renderer`.

## Phase 4: Closed Loop

The first agency milestone connects the previous pieces into one loop:

`render -> perceive -> track -> plan -> simulate -> execute -> observe -> infer`

The benchmark deliberately uses a slide-joint MuJoCo scene with three objects.
This is not a claim of general manipulation. The purpose is to test whether the
architecture can select actions from visual belief state, execute them in a real
physics backend, update physical-property beliefs, ground the result, and
replan.

Planning uses pruned 3D push primitives. Each candidate is evaluated by:

- expected mass-posterior entropy after a force observation,
- MuJoCo-predicted target displacement,
- rendered visibility after the candidate rollout.

Candidate simulation is wrapped in MuJoCo state capture and restoration, so
planning does not advance the real world. The selected action is then executed
once, and the observed velocity change updates the target object's Bayesian mass
estimator.

Plan nodes are grounded through the MuJoCo dynamics axiom. Executed action nodes
are grounded through their EFE plan, and mass updates are grounded through both
the action command and post-action observation. The restored graph is required
to have no ungrounded active nodes.

This phase intentionally leaves contact-rich free-body manipulation, rolling,
tipping, grasping, and full closed-loop MuJoCo history counterfactuals for
later work. Those need either much heavier MuJoCo rollout budgets or learned
dynamics approximations.

## Phase 5: Discoverer Slice

The first discovery implementation is intentionally narrow: friction discovery
from persistent residuals in sliding observations. It establishes the contracts
needed for learned structure proposals without claiming arbitrary concept
invention.

The discovery loop is:

1. log experiment records as replayable JSONL,
2. detect local free-energy anomalies where constant-velocity predictions fail,
3. encode anomaly features,
4. run a small trainable neural classifier constrained to a graph grammar,
5. fit the proposed friction coefficient,
6. score evidence gain against complexity and neural confidence,
7. integrate a friction property and law only above threshold,
8. audit the new concept through experiment-log observations and axioms.

The neural proposer is a deterministic one-hidden-layer model implemented in
the standard library. It currently emits only grammar-valid `friction`
proposals. The evidence gate compares squared prediction error under the old
constant-velocity model against the proposed friction model:

`v_next = sign(v) * max(0, |v| - mu * g * dt)`

False discovery prevention is part of the contract: low-error constant-velocity
logs do not produce proposals, and integrated concepts must leave the graph with
no ungrounded active nodes.

General graph transformers, new factor-type synthesis, autonomous experiment
design between competing theories, and multi-concept transfer remain future
work.

## Phase 6: Generalist Slice

The first generalization implementation is concept transfer plus causal
revision for the friction concept discovered in Phase 5. It does not treat a
learned concept as a universal constant. It stores the concept's name,
parameter, applicability signature, confidence, and revision history in a
serializable concept library.

Transfer is gated by an environment signature instead of exact material
identity. A target environment is eligible when the observed motion still has
the structural features of sliding surface interaction and measurable
deceleration. Surface identity is recorded but intentionally not required to
match, so a wood-learned friction concept can be tried on ice. Because the
material changed, the transferred target node is initialized with broad
variance and its audit remains speculative until target evidence supports a
revision.

Revision compares target observations under the transferred coefficient against
a fitted target coefficient:

`v_next = sign(v) * max(0, |v| - mu * g * dt)`

The revision is accepted only when squared-error improvement clears the
evidence threshold. Accepted revisions use the graph's normal immutable
versioning: the transferred node is superseded, incoming causal support is
copied to the replacement node, and a target observation node is linked as
revision evidence. The concept library is then updated with the new parameter
and a timestamped revision record.

Current verified benchmark: source `friction=0.30` transfers to an ice target,
target evidence revises it to `0.05`, prediction error drops from `0.21465` to
`0.0`, evidence gain exceeds `100`, and the revised node audits as confident.

## Phase 7: Historian Slice

The first historian implementation is a bounded counterfactual over experiment
methodology. It does not twin arbitrary MuJoCo histories yet. It twins a
deterministic friction/mass history where the scientific question is:

`What if friction had been discovered before mass inference?`

The replay substrate is a sequence of shared exogenous trials. Each trial fixes
the world-side variables that must be identical across actual and
counterfactual histories: trial ID, true mass, hidden friction, initial sliding
velocity, and duration. Policy-side variables are intentionally not shared:
action force, whether friction is known before mass inference, and discovery
thresholds may differ under intervention.

Each replay creates a graph-level history:

1. a policy node grounded as experiment design,
2. an initial concept-library snapshot,
3. one experiment node per trial,
4. `evolves_to` edges between consecutive experiments,
5. one mass-estimate property per experiment,
6. concept-library snapshots after each experiment,
7. a discovered friction concept when anomaly evidence clears the threshold.

Mass inference deliberately exposes the methodological failure. The world
acceleration is:

`a_observed = force / true_mass - friction * g`

If friction is not known, the replay estimates:

`mass = force / a_observed`

If friction is known, it estimates:

`mass = force / (a_observed + friction * g)`

The counterfactual intervention `discover_friction_before_mass` seeds the same
friction concept before the first experiment while keeping the exogenous trials
identical. The comparison reports mean mass-error reduction, velocity
prediction-error reduction, experiments saved before friction is available,
grounding quality, and graph size.

Current verified benchmark: six shared trials with true mass `2.5` and hidden
friction `0.25` produce actual mean mass error `0.12069` and counterfactual
mean mass error `0.0`. The actual policy discovers friction at experiment `4`;
the counterfactual policy has it at experiment `0`. Both replay graphs have no
ungrounded active nodes.

## Phase 8: Composer Slice

The first composer implementation is bounded multi-concept discovery over a
compound slide-collision observation. It does not use a graph neural network or
free-body MuJoCo playground yet. It establishes the model-comparison and graph
integration contract for concepts that interact.

The bounded grammar includes entries for friction, restitution, spring
constant, damping, and center of mass. The current benchmark actively fits only
friction and restitution:

`impact_velocity = max(0, initial_velocity - mu * g * slide_duration)`

`rebound_velocity = -e * impact_velocity`

The model selector compares:

1. friction only,
2. restitution only,
3. friction plus restitution.

Each candidate receives:

`score = evidence_gain - complexity_penalty`

where evidence gain is the log improvement over the baseline elastic,
frictionless model. This prevents the system from adding a pairwise concept
when a single concept already explains the observations.

When the pairwise model wins, Composer integrates:

1. a grounded `friction` concept node,
2. a grounded `restitution` concept node,
3. law nodes for the fitted equations,
4. a grounded relation node with interaction type
   `friction_loss_modulates_restitution`.

Current verified benchmark: eight noiseless compound observations with hidden
friction `0.25` and restitution `0.65` recover both parameters exactly. The
pairwise model scores `109.74`, beats the restitution-only and friction-only
models, adds nine graph nodes and ten edges, and leaves no ungrounded active
nodes.

## Phase 9: Theorist Slice

The first theorist implementation is bounded symbolic equation discovery. It
does not perform open-ended genetic programming, neural-guided search, or
general theorem invention. It establishes the law-discovery contract with an
auditable symbolic-regression benchmark.

Expressions are represented as trees over constants, variables, division,
powers, and square roots. Law templates expose basis expressions whose
parameters are fitted by least squares. The current pendulum search compares:

1. constant period,
2. linear length,
3. quadratic length,
4. square root of length over gravity,
5. cubic polynomial overfit candidate.

Each candidate receives:

`score = evidence_gain + scope_bonus - complexity_penalty`

where evidence gain is the log improvement over a constant baseline, complexity
penalizes larger expression trees and extra parameters, and the scope bonus
marks dimensionally consistent templates. This is enough to prefer the
square-root law for pendulum observations while preferring a simple linear law
over an overfit polynomial when the data are actually linear.

Discovered laws are not marked as axioms. They are grounded through a
symbolic-regression observation node and a theoretical-consistency node, so the
graph can audit the law while still treating it as revisable scientific
knowledge.

Current verified benchmark: twenty noiseless pendulum observations with
gravity `9.81` select `T = 6.28319*sqrt(L / g)`, recover the coefficient
`2*pi`, produce zero RMSE, score `290.49`, add three graph nodes and two edges,
and leave no ungrounded active nodes.

## Phase 10: Universalist Slice

The first universalist implementation is bounded cross-domain unification. It
does not perform arbitrary tree-edit clustering, neural-guided analogy search,
or open-ended meta-law discovery. It establishes the representation and
grounding contract for a meta-law that is supported by multiple grounded
instances and a confirmed prediction.

The current pattern is harmonic motion. Law instances are expression trees with
the structural signature:

`coefficient * sqrt(inertia / restoring)`

The benchmark supplies:

1. pendulum period: `T = 2*pi*sqrt(L/g)`,
2. spring period: `T = 2*pi*sqrt(m/k)`.

Both are integrated as grounded law-instance nodes. The Universalist abstracts
them into:

`T = 2*pi*sqrt(inertia/restoring)`

It then predicts the LC oscillator period in an unobserved domain. The correct
LC form is:

`T = 2*pi*sqrt(L*C)`

This follows the same harmonic template because inductance is the inertia-like
term and `1/C` is the restoring-like term, so `sqrt(inertia/restoring)` becomes
`sqrt(L / (1/C)) = sqrt(L*C)`.

The meta-law is not an axiom. It is grounded through incoming edges from the
grounded pendulum and spring law instances and from the tested LC prediction.
If the prediction is not tested, confidence remains below the confident audit
threshold and the meta-law is classified as speculative.

Current verified benchmark: pendulum and spring instances unify into
`harmonic_motion`, predict `T = 2*pi*sqrt(L*C)`, confirm the prediction, produce
meta-law confidence `0.90`, add seven graph nodes and six edges, and leave no
ungrounded active nodes.

## Phase 11: Strategist Slice

The first strategist implementation is bounded self-improvement over discovery
methodology. It does not perform open-ended recursive self-optimization. It
generates a small set of candidate policies, evaluates them with deterministic
Historian replay, and adopts a candidate only when it improves discovery
efficiency while preserving the broader methodology score.

The strategy policy contains:

1. action force,
2. epistemic weight,
3. anomaly threshold,
4. minimum records needed for discovery,
5. evidence and complexity weights,
6. an adoption threshold,
7. a replay horizon.

The generator currently emits three bounded perturbations:

1. `sensitive_anomaly`: lower anomaly threshold and fewer required records,
2. `stronger_actions`: higher action force,
3. `fast_discovery`: combines stronger actions with faster anomaly detection.

Each policy is converted into a Historian replay policy and evaluated on the
same shared exogenous trials. The score tracks experiments-to-friction, mean
mass error, and confident-grounding fraction. Adoption requires both:

`candidate_composite_score > baseline_composite_score`

and

`candidate_efficiency_gain >= 1 + improvement_threshold`

The adopted policy is represented as a grounded strategy-decision node with
incoming edges from replay evidence, the baseline policy score, and the
selected policy score.

Current verified benchmark: baseline discovers friction at experiment `4` with
mean mass error `0.12069`. `baseline.fast_discovery` discovers friction at
experiment `3`, reduces mean mass error to `0.05691`, preserves grounding
quality at `1.0`, is adopted with efficiency gain `1.333`, adds six graph nodes
and five edges, and leaves no ungrounded active nodes.

## Phase 12: Cartographer Slice (Facade)

The first Cartographer implementation is a bounded meta-ontology mapping substrate. It does not perform open-ended natural language semantic parsing or arbitrary multi-domain ontology alignment. It establishes the representation, traversal, and query interface for relationships between discovered physical concepts.

The Cartographer maintains a directed, attributed conceptual graph:

\[G_C = (V_C, E_C)\]

where:
- Each node \(v \in V_C\) represents a concept template (e.g., `friction`, `restitution`, `mass`) or a concrete law instance (e.g., `T = 2*pi*sqrt(L/g)`).
- Each edge \(e \in E_C\) represents a typed semantic or causal meta-relation:
  - `modulates`: a concept alters the parameter or state of another (e.g., friction modulates restitution).
  - `is_analogous_to`: a cross-domain mapping (e.g., mass in mechanics is analogous to inductance in electromagnetism, as discovered by the Universalist).
  - `contradicts`: mutually exclusive physical explanations.
  - `predicts`: a concept or law structurally implies another under specific boundary conditions.

When the Universalist unifies laws or the Composer integrates compound concepts, the Cartographer automatically updates the meta-ontology graph.

The search for distinguishing experiments uses:

1. Topological neighborhood search: identify candidate concepts that explain the observed anomaly.
2. Information-theoretic query scoring: rank experiments based on expected cross-entropy reduction between competing graph structures:
    \[I(E) = H(P(M | D)) - \mathbb{E}_{y \sim E}[H(P(M | D, y))]\]
    where \(M\) represents the candidate conceptual models, \(D\) is existing history, and \(y\) is the outcome of experiment \(E\).
3. Grounding checks: verify that any proposed edge in the conceptual map retains a complete provenance path to physical observations or domain axioms.

This phase intentionally leaves large-scale multi-domain database ingestion and open-ended natural language explanation (The Narrator) for future work.

## Phase 13: Narrator Slice (Facade)

The first Narrator implementation is a bounded scientific explanation generator. It does not perform open-ended natural language generation, creative writing, or arbitrary summarization. It establishes the contract for human-readable, auditable scientific communication from the causal graph.

The Narrator consumes the Cartographer's conceptual graph \(G_C = (V_C, E_C)\) and produces structured explanations:

\[E = (S, P)\]

where:
- Each sentence \(s \in S\) is a natural language claim about a concept, law, relationship, or experimental result.
- Each provenance path \(p \in P\) is a directed path through the causal graph from the claim to grounded roots (observations or axioms).

The generation follows a bounded template grammar instantiated by traversing \(G_C\):

| Template Slot | Cartographer Source | Grounding Requirement |
|---|---|---|
| Concept description | `ConceptNode` name, parameter, confidence | Direct link to causal graph property node |
| Law statement | `MetaRelation` with type `predicts` or law instance | Path through Theorist/Universalist nodes to observations |
| Experimental evidence | `evolves_to` chains, observation nodes | Terminal node must be observation or axiom |
| Cross-domain analogy | `MetaRelation` with type `is_analogous_to` | Both source and target concepts must be grounded |
| Contradiction resolution | `MetaRelation` with type `contradicts` | Must cite distinguishing experiment from Cartographer query |

The `generate_explanation(query_node, depth)` method:
1. Performs a bounded BFS from `query_node` in \(G_C\) to depth `depth`.
2. Collects templates for each visited node and edge type.
3. Sorts sentences by causal order: evidence \(\rightarrow\) inferences \(\rightarrow\) laws \(\rightarrow\) analogies.
4. Attaches `provenance_edges` to every generated sentence, recording the exact causal graph path that justifies the claim.

The `audit_explanation(explanation)` method verifies:
1. **Grounding completeness**: Every sentence has at least one provenance path reaching an observation or axiom.
2. **Confidence calibration**: No claim exceeds the confidence of its weakest supporting edge.
3. **Analogy validity**: Cross-domain claims only appear when `is_analogous_to` edges exist in \(G_C\).
4. **Contradiction honesty**: When competing models are discussed, the distinguishing experiment from Cartographer query is cited.
5. **Causal ordering**: Sentences appear in topological order (evidence before inference before abstraction).

This phase intentionally leaves open-ended scientific discourse, multi-agent peer review, autonomous publication, rhetorical persuasion, and natural language understanding (parsing human questions) for future work.

## Phase 14: Curator Slice (Facade)

The first Curator implementation is a bounded research question generator. It does not perform open-ended curiosity, arbitrary hypothesis invention, or unconstrained scientific speculation. It establishes the contract for an artificial scientist to identify what it does not know, prioritize gaps, and propose investigable research questions.

The Curator consumes the Narrator's explanations and the Cartographer's conceptual graph \(G_C = (V_C, E_C)\) to maintain a **frontier of ignorance**: a ranked set of open research questions.

\[F = \{q_1, q_2, ..., q_n\}\]

where each research question \(q\) is a tuple of:
- `text`: the natural language question.
- `target_concepts`: list of target concept IDs.
- `expected_information_gain` (EIG): predicted entropy reduction across \(G_C\) if answered.
- `connectivity`: measurement of graph centrality of the target concepts.
- `feasibility`: feasibility score queried from the Strategist's policy generator.
- `priority_score`: composite score weighting EIG, connectivity, and feasibility.
- `provenance_edges`: supporting path links back to explanation sentences.

The gap identification methods detect:
1. **Unexplained concept**: concept nodes with out-degree zero in `predicts` edges.
2. **Unresolved contradiction**: `contradicts` edges without a distinguishing experiment.
3. **Low-confidence knowledge**: concepts with audit confidence below threshold.
4. **Sparse neighborhood**: concepts with degree below threshold \(k\).
5. **Anomalous residual**: explanations marked with unexplained variance or low belief confidence.

The `propose_research_agenda(top_k)` method:
1. Identifies gaps from the Narrator's explanation.
2. Scores and ranks candidate questions.
3. For each top-ranked question, queries the Cartographer for distinguishing experiments and the Strategist for policies to execute them.
4. Returns a `ResearchAgenda` containing the questions, proposed experiments, and policies.

The `audit_agenda(agenda)` method verifies:
1. Every proposed experiment has a grounded Cartographer query.
2. Every policy has a feasible Strategist evaluation.
3. No question targets an already-resolved contradiction.
4. All questions are grounded through the Narrator's explanation provenance.

This phase intentionally leaves open-ended curiosity, value-driven research prioritization (e.g., "important" vs. "interesting"), multi-agent debate over agendas, and autonomous resource allocation for future work.

## Phase 15: Collaborator Slice (Facade)

The first Collaborator implementation is a bounded scientific debate engine. It does not perform open-ended social simulation, arbitrary persuasion, or unconstrained multi-agent negotiation. It establishes the contract for an artificial scientist to argue competing hypotheses, identify evidence that would resolve disagreement, and record consensus or dissent as grounded meta-meta-knowledge.

The Collaborator instantiates multiple **internal scientist agents** (`ScientistAgent`) from a single causal graph, each with a distinct prior over a contested subgraph:

\[Agent_i = (G, prior_i, belief_i, narrator_i)\]

where:
- \(G\) is the shared causal graph.
- \(prior_i\) is a perturbed initial belief over the contested variables.
- \(belief_i\) is the agent's current posterior after conditioning on evidence.
- \(narrator_i\) is a Narrator instance bound to the agent's belief state.

The debate follows structured scientific discourse:
1. **Position formation**: Each agent uses its Narrator to generate a grounded explanation of the contested phenomenon.
2. **Disagreement identification**: The Cartographer compares explanations and identifies `contradicts` edges between competing claims.
3. **Evidence weighing**: Each agent audits the other's explanation. Disagreements are classified as: empirical, theoretical, methodological, or irreducible.
4. **Experiment proposal**: For empirical disagreements, the Curator proposes distinguishing experiments via the Cartographer.
5. **Execution and resolution**: The selected experiment executes (Agency), results update the shared graph, and agents revise beliefs.
6. **Consensus recording**: Convergence adds a `consensus` meta-meta-edge to the Cartographer's ontology. Irreducible conflict adds a `dissent` edge with full provenance.

The `Collaborator.debate(contested_node, agent_priors, max_rounds)` method:
1. Initializes agents with perturbed priors over the contested subgraph.
2. For each round:
   a. Agents generate explanations.
   b. Cartographer identifies contradictions.
   c. Agents audit each other's explanations.
   d. Curator proposes distinguishing experiments.
   e. Feasible experiments execute, and shared graph updates.
   f. Agents update beliefs. Convergence halts the loop.
3. Returns a `DebateRecord` logging all rounds, final beliefs, and consensus/dissent edges.

The `audit_debate(record)` method verifies:
1. Every agent explanation was fully grounded at generation.
2. Every contradiction identified existed in \(G_C\) at debate start.
3. Every executed experiment had a grounded distinguishing query.
4. Every consensus or dissent edge has complete provenance.
5. No agent had privileged access to evidence.

This phase intentionally leaves open-ended social dynamics, rhetorical persuasion, authority-based trust, emotional negotiation, and human-in-the-loop debate for future work.

## Phase 16: Integrator Slice (Orchestrator)

The first Integrator implementation is a bounded cognitive loop orchestrator. It does not perform open-ended cognitive architecture synthesis. It establishes the sequence and execution rules for coordinating all previous cognitive subsystems into a single lifecycle.

The Integrator coordinates the execution cycle:

\[observe \rightarrow learn \rightarrow generalize \rightarrow prioritize \rightarrow debate \rightarrow act\]

where:
1. **Agency & Perception**: Runs push actions in the simulator to gather tracking traces.
2. **Composer & Discoverer**: Detects anomalies and proposes/integrates concepts.
3. **Universalist Abstraction**: Abstract and unifies laws into cross-domain principles.
4. **Strategist Policy**: Evaluates policies over replay histories and adopts improved discovery methodologies.
5. **Curator Gap Detection**: Maps the conceptual topology to discover unexplained gaps and prioritize them in a `ResearchAgenda`.
6. **Collaborator Discourse**: Spawns competing scientist agents to debate conflicting claims.

The Integrator wires the simulator directly into the debate-resolving loop: when the Collaborator debates, instead of mocking updates, the Integrator triggers actual simulator rollouts based on the Curator's distinguishing experiments. The physical results of these actions are used to update the agents' beliefs.

The `run_orchestration_cycle(target_concept)` method:
1. Runs closed-loop active inference steps.
2. Proposes and integrates concepts into the ontology graph via Composer.
3. Unifies discovered laws into cross-domain patterns via Universalist.
4. Assesses methodology score and adopts policies via Strategist.
5. Scans ontology graph for sparse gaps and prioritizes open research questions.
6. Conducts multi-agent debates backed by simulator rollouts to resolve empirical disputes, and writes a grounded `consensus` edge.

The `IntegratorResult` audit verifies:
1. Every stage of the orchestration loop executed successfully.
2. Grounding quality achieves `1.0` with `confident` audit status.
3. Final ontology contains all integrated concepts, meta-laws, and consensus relation edges.



