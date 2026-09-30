(define (domain wumpus-classical)
  (:requirements :strips :typing)
  (:types location)
  (:predicates
    (at ?x - location)
    (safe ?x - location)
    (adjacent ?x - location ?y - location)
    (gold-at ?x - location)
    (have-gold)
    (climbed-out)
  )

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?from) (safe ?to) (adjacent ?from ?to))
    :effect (and (not (at ?from)) (at ?to)))

  (:action grab
    :parameters (?where - location)
    :precondition (and (at ?where) (gold-at ?where))
    :effect (and (not (gold-at ?where)) (have-gold)))

  (:action climb
    :parameters (?where - location)
    :precondition (and (at ?where) (have-gold))
    :effect (climbed-out))
)
