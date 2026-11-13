# ProcureBench

> Can an AI agent keep a school kitchen supplied when the world refuses to stay simple?

Our Junction 2026 project for **AfterQuery — Break the Agent**. ProcureBench is a reproducible reinforcement-learning environment where an agent plans food procurement for a school kitchen under budget, inventory, delivery, nutrition, allergen, and supply-disruption constraints.

## Why this environment matters

Real-world agents must make sequential decisions under uncertainty. A cheap-looking action can cause a missed delivery tomorrow; a substitute can violate an allergy restriction; a high reward can be obtained by exploiting a weak metric instead of doing the job.

ProcureBench makes those failures measurable. The agent gets the information and tools needed to succeed—but has to reason across time, competing objectives, and hidden failure modes.

## The task

The agent manages a five-day kitchen plan. It must ensure every planned meal can be served safely and on time while keeping cost and waste low.

~~~
Read menu, dietary requirements, stock, supplier offers, and delivery windows
        ↓
Search products / inspect substitutions / place or amend orders
        ↓
Handle stock-outs, late deliveries, price changes, and demand changes
        ↓
Serve each day safely, on budget, and with minimal waste
~~~

## What an agent can do

- Inspect inventory, menus, budgets, supplier offers, and delivery schedules.
- Search products and their nutrition/allergen attributes.
- Place, amend, cancel, or reserve orders.
- Choose approved substitutions.
- Ask for an approval when policy requires it.
- Advance simulated time and observe consequences.

## What makes it hard

- **Delayed consequences:** an order may look good now but arrive too late.
- **Partial observability:** suppliers can change stock, prices, and reliability.
- **Hard safety constraints:** allergens and nutrition cannot be traded for cost.
- **Coupled decisions:** one ingredient serves multiple future meals.
- **Trade-offs:** cost, waste, delivery risk, and quality compete.
- **Adversarial shortcuts:** agents may try to avoid purchases, exploit accounting, or claim meals were served without sufficient ingredients.

## Reward design

The reward represents real outcome, not superficial activity:

~~~
+ meals safely served and nutritionally compliant
- missed or unsafe meals                    (large penalty)
- allergen violations                       (terminal / very large penalty)
- budget overruns
- expired or wasted inventory
- late deliveries and unnecessary approvals
~~~

The environment independently validates meal service from the actual inventory and recipe requirements. An agent cannot gain reward by merely reporting success or placing meaningless orders.

## How we evaluate agents

We will run:

1. a frontier agent using the provided AfterQuery harness;
2. a transparent reference planner (constraint-aware heuristic / solver);
3. multiple fixed random seeds and disruption scenarios;
4. targeted reward-hacking attempts.

We will report meal-service rate, safety violations, cost, waste, reward, and reproducibility. The point is not simply who has the highest score—it is identifying **why** an agent fails.

## Reference cases

| Scenario | Tests |
| --- | --- |
| Normal week | Basic planning and inventory management |
| Milk stock-out | Substitution and delivery reasoning |
| Allergen conflict | Hard safety constraints |
| Last-minute event | Demand spike and deadline management |
| Price shock | Cost-versus-risk trade-off |
| Reward-hacking probe | Whether reported success matches physical inventory truth |

## Architecture

~~~
Scenario generator + deterministic simulator
        ↓
Structured observation and action API
        ↓
AfterQuery agent harness / frontier agent
        ↔ tools: search, inspect, order, amend, approve, advance time
        ↓
Independent validator + reward engine
        ↓
Metrics, traces, replay, and comparison dashboard
~~~

## Planned stack

| Area | Choice | Why |
| --- | --- | --- |
| Environment | Python | Deterministic simulation and testing |
| API/harness | FastAPI + structured JSON tools | Clear agent observation/action contract |
| Data | MariaDB | Runs, episode traces, outcomes, and comparisons |
| Compute | Modal or Verda | Parallel evaluations and optional post-training |
| Models | AfterQuery-provided models; Gemini/Claude optional | Frontier evaluation and analysis |
| UI | Next.js / React | Live episode replay and score comparison |
| Storage | IronShard | Datasets, traces, checkpoints, and artifacts |
| CI | CRACI | Reproducible tests and build provenance |

## First milestones

- [ ] Define state, action, observation, termination, and reward contracts.
- [ ] Build a deterministic one-day inventory/order simulator.
- [ ] Add recipes, allergens, quantities, supplier offers, and delivery windows.
- [ ] Add one disruption: supplier stock-out.
- [ ] Add independent meal-service validation.
- [ ] Build a simple baseline heuristic.
- [ ] Connect the AfterQuery harness and run a frontier agent.
- [ ] Add fixed seeds, episode logging, and replay.
- [ ] Add explicit reward-hacking tests.
- [ ] Expand to a five-day environment only after the one-day loop works.

## Team roles

| Role | Focus |
| --- | --- |
| Environment/reward | Simulator, state transitions, validator, anti-hacking tests |
| Agent/research | Harness integration, baseline, experiment design, analysis |
| Backend/data | APIs, traces, run storage, reproducibility |
| Frontend/demo | Replay dashboard, metrics, presentation |

## Definition of a winning demo

In under three minutes, show a frontier agent make a credible but costly mistake, replay exactly why it failed, then compare it with a reproducible reference planner. End with a reward-hacking attempt that the environment catches.

## Non-negotiable principles

- A high reward must mean the kitchen really served safe meals.
- Every episode must be replayable from a seed.
- The agent must have enough information and tools to succeed.
- Failures should reveal meaningful reasoning/planning limits, not missing API access.
- Keep the first version small, deterministic, and hard to game.

## Prize entries

Main entry: **Break the Agent — AfterQuery**.

Potential side entries when visibly used: Best Use of Modal or Verda, Codex, MariaDB/Aiven, CRACI, and IronShard.

## Local setup

This is currently a project kickoff. Environment setup instructions will be added with the first runnable simulator.
