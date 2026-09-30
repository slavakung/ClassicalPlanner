(define (domain tiny-grid)
  (:requirements :strips :typing :negative-preconditions :action-costs)
  (:types location)
  (:predicates
    (at ?x - location)
    (adjacent ?from - location ?to - location)
    (blocked ?x - location)
  )
  (:functions (total-cost))

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and
      (at ?from)
      (adjacent ?from ?to)
      (not (blocked ?to)))
    :effect (and
      (not (at ?from))
      (at ?to)
      (increase (total-cost) 1)))
)
