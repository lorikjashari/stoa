# Aimo Agent Gateway

> Safe, explainable B2B food purchasing for autonomous agents.

Our Junction 2026 project for **Valio Aimo — Trusted Trade Agents**. We are prototyping the trusted interface through which an external purchasing agent can discover products, check price and availability, propose an order, and request human approval when needed.

## The problem

Professional kitchens order thousands of products under time, budget, dietary, and delivery constraints. Future buyer agents can save time, but Valio Aimo must not expose confidential systems or let an agent make uncontrolled commitments.

Today's webshop is built for people clicking a catalogue. We need a safe operating model for **agents acting on behalf of people**.

## Our solution

**Aimo Agent Gateway** is a mock structured API plus a human-friendly control centre.

```text
Kitchen need: “I need 1,000 litres of lactose-free milk by Monday.”
        ↓
Agent searches the catalogue and checks price + availability
        ↓
Gateway applies authority, budget, allergen, delivery, and substitution rules
        ↓
Agent either creates an order or sends the best option to a human for approval
        ↓
Every decision is recorded with a clear, auditable explanation
```

## Core capabilities

1. **Product, price, and availability lookup** — structured search results for an external purchasing agent.
2. **Order proposal and placement** — move safely from enquiry to a fixed-price commitment.
3. **Permission and approval rules** — decide what an agent can do alone and when a human must approve.
4. **Explainable audit trail** — show what the agent chose, why, which rules applied, and who approved it.
5. **Safe substitutions** — rank alternatives while respecting allergies, nutrition needs, preferences, and budget.

## Our winning demo scenario

A school kitchen needs lactose-free milk for Monday. The preferred product is short on stock.

The agent:

1. finds eligible lactose-free alternatives;
2. compares price, availability, delivery date, and allergen data;
3. proposes the best substitute;
4. sees that the substitution crosses a policy threshold;
5. requests a manager's approval instead of ordering automatically;
6. shows a readable record of the decision after approval.

This demonstrates the real problem: useful autonomy **without losing control**.

## Why this is different

- **Not a chatbot webshop:** we define an agent-to-business interaction model.
- **Trust is enforceable:** every action passes structured policy checks.
- **Human approval is built in:** agents can be helpful without unlimited authority.
- **Decisions are explainable:** buyer and supplier can see the reason for each order or exception.
- **Food-service aware:** substitutions consider availability, delivery, allergens, nutrition, and commercial rules.

## Proposed architecture

```text
Buyer agent / demo UI
        ↓
Agent Gateway API
  ├─ catalogue search: products, price, availability
  ├─ policy engine: authority, budget, allergen, substitution rules
  ├─ order workflow: draft → approval required / approved → placed
  └─ audit ledger: decision, evidence, policy result, approver
        ↓
Mock Valio Aimo catalogue and order backend
        ↓
Human approval dashboard + order/audit timeline
```

## Planned stack

| Area | Choice | Why |
| --- | --- | --- |
| UI | Next.js / React | Agent and manager views in one clear demo |
| Backend | Python + FastAPI | Structured API, policy engine, and testable workflows |
| Data | MariaDB | Product catalogue, orders, approval rules, audit records |
| AI reasoning | Gemini or Claude | Turn needs into structured requirements and explain trade-offs |
| Deployment | Modal, Verda, or Aiven | Reliable public demo and agent workloads |
| Live enrichment | Context.dev or Exa *(optional)* | Verify product/certification information when appropriate |
| CI/provenance | CRACI | Secure, reproducible builds |

## First milestones

- [ ] Get mock catalogue/API materials and ask Valio Aimo mentors about constraints.
- [ ] Choose one complete scenario: lactose-free milk substitution for a school kitchen.
- [ ] Create catalogue, availability, allergen, price, and delivery mock data.
- [ ] Implement product search and structured purchase request.
- [ ] Implement policy rules: budget, approval threshold, allergen safety, substitutions.
- [ ] Build draft order → approval → placed workflow.
- [ ] Build an audit timeline a non-technical manager can understand.
- [ ] Test happy path, partial availability, unsafe substitute, and rejected order.

## Team roles

| Role | Focus |
| --- | --- |
| Product/demo | Scenario, policies, demo story, pitch |
| Backend | API contract, catalogue, policy engine, order workflow |
| AI/agent | Requirement extraction, tool calling, explanations |
| Frontend | Search/proposal view, approval dashboard, audit timeline |
| Platform | Database, deploy, tests, CI |

## Demo success criteria

In under two minutes, a judge should see:

1. an agent retrieve structured price and availability data;
2. a policy-safe decision about an order or substitute;
3. a human approval request when the agent crosses its authority;
4. a completed order or a safe refusal;
5. a clear audit record explaining what happened and why.

## Non-negotiable rules

- Never let the agent order outside its granted authority.
- Never suggest an allergen-unsafe substitute.
- Never hide uncertainty, stock limitations, or delivery trade-offs.
- Every commitment must have an audit event and a human-readable reason.
- Use synthetic/mock data only unless Valio Aimo provides cleared data.

## Prize entries

Main entry: **Trusted Trade Agents — Valio Aimo**.

We can also enter Best Use of Gemini, Modal or Verda, Codex, Aiven, CRACI, Context.dev, and Exa only when each tool has a visible, meaningful role in the working flow.

## Local setup

This repository is currently a project kickoff. Setup instructions will be added once the frontend and API are scaffolded.

## Useful links

- [Junction 2026](https://hackjunction.com/)
