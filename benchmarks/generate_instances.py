#!/usr/bin/env python3
"""Deterministically generate literature-inspired STRIPS planning instances.

Only the Python standard library is required. --check verifies all fixture
bytes and their manifest hashes. Only generated fixtures are distributed.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks"
REVISION = "e21d49c2cb61d147a46c5966f2581bf6fd422b9f"
UPSTREAM = "https://github.com/aibasel/downward-benchmarks"
REFERENCES = {
    "blocks": "https://www.cs.umd.edu/~nau/plasma/automated-planning.html",
    "gripper": UPSTREAM + "/tree/" + REVISION + "/gripper",
    "logistics": UPSTREAM + "/tree/" + REVISION + "/logistics00",
    "hanoi": "https://www.cs.cmu.edu/~cburch/survey/recurse/hanoi.html",
    "sliding_tile": "https://aima.cs.berkeley.edu/lisp/doc/overview-SEARCH.html",
}

BLOCKS = """(define (domain blocks-generated)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (ontable ?x - block)
    (clear ?x - block) (holding ?x - block) (handempty))
  (:action pickup :parameters (?x - block)
    :precondition (and (ontable ?x) (clear ?x) (handempty))
    :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty))))
  (:action putdown :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))
  (:action stack :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))))
  (:action unstack :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))))
"""

GRIPPER = """(define (domain gripper-generated)
  (:requirements :strips :typing)
  (:types room ball gripper)
  (:predicates (robot-at ?r - room) (at ?b - ball ?r - room)
    (free ?g - gripper) (carry ?b - ball ?g - gripper) (connected ?r - room ?s - room))
  (:action move :parameters (?from - room ?to - room)
    :precondition (and (robot-at ?from) (connected ?from ?to))
    :effect (and (robot-at ?to) (not (robot-at ?from))))
  (:action pick :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (robot-at ?r) (at ?b ?r) (free ?g))
    :effect (and (carry ?b ?g) (not (at ?b ?r)) (not (free ?g))))
  (:action drop :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (robot-at ?r) (carry ?b ?g))
    :effect (and (at ?b ?r) (free ?g) (not (carry ?b ?g)))))
"""

HANOI = """(define (domain hanoi-generated)
  (:requirements :strips :typing)
  (:types support - object disk peg - support)
  (:predicates (on ?d - disk ?s - support) (clear ?s - support)
    (smaller ?d - disk ?s - support))
  (:action move :parameters (?d - disk ?from - support ?to - support)
    :precondition (and (on ?d ?from) (clear ?d) (clear ?to) (smaller ?d ?to))
    :effect (and (on ?d ?to) (clear ?from) (not (on ?d ?from)) (not (clear ?to)))))
"""

SLIDING = """(define (domain sliding-generated)
  (:requirements :strips :typing)
  (:types tile cell)
  (:predicates (at ?t - tile ?c - cell) (blank ?c - cell)
    (adjacent ?a - cell ?b - cell))
  (:action slide :parameters (?t - tile ?from - cell ?to - cell)
    :precondition (and (at ?t ?from) (blank ?to) (adjacent ?from ?to))
    :effect (and (at ?t ?to) (blank ?from) (not (at ?t ?from)) (not (blank ?to)))))
"""

LOGISTICS = """(define (domain logistics-generated)
  (:requirements :strips :typing)
  (:types location package truck airplane - object airport depot - location)
  (:predicates (package-at ?p - package ?l - location) (truck-at ?t - truck ?l - location)
    (plane-at ?a - airplane ?l - airport) (in-truck ?p - package ?t - truck)
    (in-plane ?p - package ?a - airplane) (road ?x - location ?y - location)
    (flight ?x - airport ?y - airport))
  (:action load-truck :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (package-at ?p ?l) (truck-at ?t ?l))
    :effect (and (in-truck ?p ?t) (not (package-at ?p ?l))))
  (:action unload-truck :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (in-truck ?p ?t) (truck-at ?t ?l))
    :effect (and (package-at ?p ?l) (not (in-truck ?p ?t))))
  (:action drive :parameters (?t - truck ?from - location ?to - location)
    :precondition (and (truck-at ?t ?from) (road ?from ?to))
    :effect (and (truck-at ?t ?to) (not (truck-at ?t ?from))))
  (:action load-plane :parameters (?p - package ?a - airplane ?l - airport)
    :precondition (and (package-at ?p ?l) (plane-at ?a ?l))
    :effect (and (in-plane ?p ?a) (not (package-at ?p ?l))))
  (:action unload-plane :parameters (?p - package ?a - airplane ?l - airport)
    :precondition (and (in-plane ?p ?a) (plane-at ?a ?l))
    :effect (and (package-at ?p ?l) (not (in-plane ?p ?a))))
  (:action fly :parameters (?a - airplane ?from - airport ?to - airport)
    :precondition (and (plane-at ?a ?from) (flight ?from ?to))
    :effect (and (plane-at ?a ?to) (not (plane-at ?a ?from)))))
"""


def atom(*words):
    return "(" + " ".join(words) + ")"


def problem(name, domain, objects, initial, goal):
    return (f"(define (problem {name})\n  (:domain {domain})\n"
            f"  (:objects {objects})\n  (:init {' '.join(initial)})\n"
            f"  (:goal (and {' '.join(goal)})))\n")


def ground_domain(name, predicates, actions):
    lines = [f"(define (domain {name})", "  (:requirements :strips)",
             "  (:predicates " + " ".join(f"({p})" for p in predicates) + ")"]
    for action, pre, add, delete in actions:
        lines.extend([f"  (:action {action} :parameters ()",
                      "    :precondition (and " + " ".join(atom(p) for p in pre) + ")",
                      "    :effect (and " + " ".join([atom(p) for p in add] +
                        ["(not " + atom(p) + ")" for p in delete]) + "))"])
    return "\n".join(lines) + ")\n"


def sliding_oracle():
    """Exhaustively solve the 181440-state reachable 8-puzzle component.

    Independent of PDDL and C++: plain tuples, swaps, deque and distances.
    First discovery at each requested depth fixes reproducible sample states.
    """
    adjacency = [[j for j in range(9)
                  if abs(i // 3 - j // 3) + abs(i % 3 - j % 3) == 1]
                 for i in range(9)]
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    distance = {goal: 0}
    queue = deque([goal])
    chosen = {}
    while queue:
        board = queue.popleft()
        depth = distance[board]
        chosen.setdefault(depth, board)
        blank = board.index(0)
        for other in adjacency[blank]:
            successor = list(board)
            successor[blank], successor[other] = successor[other], successor[blank]
            successor = tuple(successor)
            if successor not in distance:
                distance[successor] = depth + 1
                queue.append(successor)
    assert len(distance) == 181440 and max(distance.values()) == 31
    return adjacency, chosen


def build():
    files, rows = {}, []

    def put(relative, contents):
        files["benchmarks/instances/generated/" + relative] = contents.encode("utf-8")
        return "benchmarks/instances/generated/" + relative

    def register(identifier, family, domain, task, size, optimum, notes,
                 representation="typed_first_order", difficulty="medium", **extra):
        rows.append(dict(id=identifier, family=family, representation=representation,
                         domain=domain, problem=task, size=size,
                         source_kind="generated", reference=REFERENCES[family],
                         optimal_length=optimum, notes=notes, difficulty=difficulty, **extra))

    blocks_domain = put("blocks/domain.pddl", BLOCKS)
    initial = ["(on c a)", "(ontable a)", "(ontable b)", "(clear c)",
               "(clear b)", "(handempty)"]
    task = put("blocks/sussman.pddl", problem("sussman", "blocks-generated",
               "a b c - block", initial, ["(on a b)", "(on b c)"]))
    register("blocks-sussman", "blocks", blocks_domain, task, 3, 6,
             "Sussman anomaly: initially c on a; goal a on b on c. Four primitive arm actions.",
             difficulty="small", optimality_basis="six necessary pickup/unstack and stack/putdown actions")
    for n in (3, 4, 5, 6, 7, 8):
        names = [f"b{i}" for i in range(1, n + 1)]
        initial = [atom("on", names[i], names[i+1]) for i in range(n - 1)]
        initial += [atom("ontable", names[-1]), atom("clear", names[0]), "(handempty)"]
        goal = [atom("on", names[i+1], names[i]) for i in range(n - 1)]
        goal += [atom("ontable", names[0]), "(handempty)"]
        task = put(f"blocks/reverse-{n:02}.pddl", problem(f"blocks-reverse-{n}",
                   "blocks-generated", " ".join(names) + " - block", initial, goal))
        register(f"blocks-reverse-{n:02}", "blocks", blocks_domain, task, n, 2*n,
                 "Reverse one tower; every block must be lifted and placed once.",
                 difficulty="small" if n <= 4 else "hard" if n >= 7 else "medium",
                 optimality_basis="2*n lower bound attained by successive placement into reversed tower")

    gripper_domain = put("gripper/domain.pddl", GRIPPER)
    for n in (2, 3, 4, 5, 6, 8, 10):
        balls = [f"b{i}" for i in range(1, n + 1)]
        initial = ["(robot-at ra)", "(free left)", "(free right)",
                   "(connected ra rb)", "(connected rb ra)"]
        initial += [atom("at", b, "ra") for b in balls]
        goal = [atom("at", b, "rb") for b in balls]
        task = put(f"gripper/balls-{n:02}.pddl", problem(f"gripper-{n}", "gripper-generated",
                   "ra rb - room left right - gripper " + " ".join(balls) + " - ball", initial, goal))
        register(f"gripper-{n:02}", "gripper", gripper_domain, task, n, 2*n + 2*((n+1)//2)-1,
                 "Two grippers, two rooms, all balls move from ra to rb; robot finish unconstrained.",
                 difficulty="small" if n <= 3 else "hard" if n >= 8 else "medium",
                 optimality_basis="2*n pickups/drops plus 2*ceil(n/2)-1 room traversals")

    hanoi_domain = put("hanoi/domain.pddl", HANOI)
    for n in (2, 3, 4, 5, 6, 7, 8, 10):
        disks, pegs = [f"d{i}" for i in range(1, n+1)], ["pa", "pb", "pc"]
        supports = disks + pegs
        initial = [atom("on", disks[i], disks[i+1]) for i in range(n-1)]
        initial += [atom("on", disks[-1], "pa"), atom("clear", disks[0]), "(clear pb)", "(clear pc)"]
        static = [atom("smaller", d, s) for i, d in enumerate(disks) for s in disks[i+1:] + pegs]
        goal = [atom("on", disks[i], disks[i+1]) for i in range(n-1)] + [atom("on", disks[-1], "pc")]
        task = put(f"hanoi/disks-{n:02}.pddl", problem(f"hanoi-{n}", "hanoi-generated",
                   " ".join(disks) + " - disk pa pb pc - peg", initial + static, goal))
        info = dict(pair_id=f"hanoi-{n:02}", optimality_basis="T(n)=2*T(n-1)+1, T(0)=0",
                    reachable_states=3**n, difficulty="small" if n <= 3 else "hard" if n >= 8 else "medium")
        register(f"hanoi-fo-{n:02}", "hanoi", hanoi_domain, task, n, 2**n-1,
                 "Three pegs; d1 smallest. Typed disk/peg subtypes of support.", **info)
        predicates = [f"on-{d}-{s}" for i, d in enumerate(disks) for s in disks[i+1:] + pegs]
        predicates += [f"clear-{s}" for s in supports]
        actions = []
        for i, d in enumerate(disks):
            for source in disks[i+1:] + pegs:
                for target in disks[i+1:] + pegs:
                    if source != target:
                        actions.append((f"move-{d}-{source}-{target}",
                            [f"on-{d}-{source}", f"clear-{d}", f"clear-{target}"],
                            [f"on-{d}-{target}", f"clear-{source}"],
                            [f"on-{d}-{source}", f"clear-{target}"]))
        domain = put(f"hanoi/propositional-{n:02}-domain.pddl", ground_domain(f"hanoi-prop-{n}", predicates, actions))
        def flatten(facts):
            return [atom(fact[1:-1].replace(" ", "-")) for fact in facts]
        task = put(f"hanoi/propositional-{n:02}.pddl", problem(f"hanoi-prop-{n}", f"hanoi-prop-{n}",
                   "", flatten(initial), flatten(goal)))
        register(f"hanoi-prop-{n:02}", "hanoi", domain, task, n, 2**n-1,
                 "Zero-argument predicates/actions; static size checks compiled away; same reachable graph as typed pair.",
                 representation="propositional", **info)

    sliding_domain = put("sliding_tile/domain.pddl", SLIDING)
    adjacency, boards = sliding_oracle()
    predicates = [f"at-t{tile}-c{cell}" for tile in range(1, 9) for cell in range(9)]
    predicates += [f"blank-c{cell}" for cell in range(9)]
    actions = [(f"slide-t{tile}-c{source}-c{target}",
                [f"at-t{tile}-c{source}", f"blank-c{target}"],
                [f"at-t{tile}-c{target}", f"blank-c{source}"],
                [f"at-t{tile}-c{source}", f"blank-c{target}"])
               for tile in range(1, 9) for source in range(9) for target in adjacency[source]]
    prop_domain = put("sliding_tile/propositional-domain.pddl", ground_domain("sliding-prop", predicates, actions))
    for distance in (4, 6, 10, 14, 18, 24, 31):
        board = boards[distance]
        goal = [atom("at", f"t{t}", f"c{t-1}") for t in range(1, 9)] + ["(blank c8)"]
        initial = [atom("at", f"t{t}", f"c{i}") if t else atom("blank", f"c{i}")
                   for i, t in enumerate(board)]
        static = [atom("adjacent", f"c{i}", f"c{j}") for i in range(9) for j in adjacency[i]]
        objects = " ".join(f"t{i}" for i in range(1, 9)) + " - tile "
        objects += " ".join(f"c{i}" for i in range(9)) + " - cell"
        task = put(f"sliding_tile/distance-{distance:02}.pddl", problem(f"sliding-{distance}",
                   "sliding-generated", objects, initial + static, goal))
        info = dict(pair_id=f"sliding-{distance:02}", initial_board=list(board),
                    optimality_basis="independent exhaustive reverse BFS over all 181440 reachable tuple states",
                    reachable_states=181440, difficulty="small" if distance <= 6 else "hard" if distance >= 24 else "medium")
        register(f"sliding-fo-{distance:02}", "sliding_tile", sliding_domain, task, distance, distance,
                 "3x3 sliding puzzle; size is exact goal distance, not number of tiles; blank encoded as 0.", **info)
        task = put(f"sliding_tile/propositional-{distance:02}.pddl", problem(f"sliding-prop-{distance}",
                   "sliding-prop", "", flatten(initial), flatten(goal)))
        register(f"sliding-prop-{distance:02}", "sliding_tile", prop_domain, task, distance, distance,
                 "Fully grounded zero-argument encoding of the identical 8-puzzle graph.",
                 representation="propositional", **info)

    logistics_domain = put("logistics/domain.pddl", LOGISTICS)
    for n in (1, 2, 3, 4):
        packages = [f"p{i}" for i in range(1, n+1)]
        initial = ["(truck-at ta da)", "(truck-at tb db)", "(plane-at plane aa)",
                   "(road da aa)", "(road aa da)", "(road db ab)", "(road ab db)",
                   "(flight aa ab)", "(flight ab aa)"] + [atom("package-at", p, "da") for p in packages]
        goal = [atom("package-at", p, "db") for p in packages]
        task = put(f"logistics/packages-{n:02}.pddl", problem(f"logistics-{n}", "logistics-generated",
                   "da db - depot aa ab - airport ta tb - truck plane - airplane " + " ".join(packages) + " - package",
                   initial, goal))
        register(f"logistics-{n:02}", "logistics", logistics_domain, task, n, 6*n+4,
                 "Two-city depot-airport relay; unlimited capacities; every package uses two trucks and one plane.",
                 difficulty="small" if n == 1 else "hard" if n >= 4 else "medium",
                 optimality_basis="6*n mandatory transfers plus 1 source-truck drive, 1 flight and 2 destination-truck drives")

    for row in rows:
        for kind in ("domain", "problem"):
            row[kind + "_sha256"] = hashlib.sha256(files[row[kind]]).hexdigest()
    manifest = dict(schema_version=1, generator="benchmarks/generate_instances.py",
                    units="unit-cost primitive sequential actions", instances=rows)
    files["benchmarks/manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify reproducibility without modifying files")
    args = parser.parse_args()
    files, rows = build()
    for relative, data in files.items():
        path = ROOT / relative
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Fixture does not match generator: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    verb = "Verified" if args.check else "Generated"
    print(f"{verb} {len(rows)} instances in {len(set(row['family'] for row in rows))} families; {len(files)-1} PDDL files.")


if __name__ == "__main__":
    main()
