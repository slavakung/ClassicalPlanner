# Supported PDDL subset (revision 0.4.0)

The implementation is intentionally strict.  Unsupported syntax is rejected
rather than approximated.

## Supported

- `define`, `domain`, `problem`
- `:requirements`
  - `:strips`
  - `:typing`
  - `:negative-preconditions`
  - `:equality`
  - `:action-costs`
  - `:adl` declarations when the actual expressions use the supported subset;
    the declaration does not enable disjunction, quantifiers or conditional effects
- `:types` with a simple inheritance chain/tree
- `:constants`
- `:objects`
- `:predicates`
- `:action`
  - typed `:parameters`
  - conjunction of positive/negative literal `:precondition`s
  - conjunction of positive/negative literal `:effect`s
  - `(increase (total-cost) N)` and immutable numeric function applications,
    e.g. `(increase (total-cost) (distance ?from ?to))`
- `:functions` containing `total-cost` and typed static numeric functions;
  optional `- number` return annotations are supported
- `:init`
  - positive ground facts
  - `(= (total-cost) N)`
  - numeric initializations such as `(= (distance a b) 3)`
- `:goal`
  - conjunction of positive/negative ground literals
  - ground equality
- `(:metric minimize (total-cost))`
- semicolon line comments
- case-insensitive PDDL symbols (normalized to lowercase)

The runtime uses the PDDL closed-world assumption: an unlisted ground atom is
false.  Equality is interpreted and is not stored as a state fluent.

Static numeric functions are evaluated while grounding. Each function application
used by an emitted action must be initialized; missing values are rejected, not
assumed to be zero. Duplicate initializations, negative cost increases, and costs
that overflow or underflow the runtime float representation are rejected. Multiple
cost increases in one action are summed. Numeric functions cannot be modified,
tested in conditions, or used in arithmetic expressions in this subset.

Grounding identifies predicates that no action changes, joins their positive
initial-state relations using argument indexes, and prunes bindings that violate
static negative preconditions or equality. This avoids enumerating the full typed
Cartesian product for sparse relational domains. Remaining unconstrained parameters
are still grounded eagerly. Static preconditions are retained in the resulting
actions, and surviving actions retain schema order and lexicographic object-binding
order. Removing impossible actions can change numeric action IDs relative to older
builds; plans must be replayed against the matching problem/build.

## Not yet supported

- disjunction
- existential/universal quantification
- conditional effects (`when`)
- derived predicates
- arbitrary numeric fluents and numeric expressions
- durative actions / temporal PDDL
- preferences and constraints
- probabilistic effects
- lazy grounding
- lifted/unification-based search

`ExecutionMode::lazyGround` and `ExecutionMode::lifted` are reserved and throw
until implemented.  `Ast.h` includes reserved numeric/temporal node shapes for a
future PDDL 2.1 extension.
