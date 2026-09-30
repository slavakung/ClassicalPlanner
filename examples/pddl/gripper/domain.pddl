(define (domain gripper-mini)
  (:requirements :strips :typing)
  (:types room ball gripper)
  (:predicates
    (robot-at ?r - room)
    (ball-at ?b - ball ?r - room)
    (free ?g - gripper)
    (carry ?b - ball ?g - gripper)
  )

  (:action move
    :parameters (?from - room ?to - room)
    :precondition (robot-at ?from)
    :effect (and (not (robot-at ?from)) (robot-at ?to)))

  (:action pick
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (robot-at ?r) (ball-at ?b ?r) (free ?g))
    :effect (and (not (ball-at ?b ?r)) (not (free ?g)) (carry ?b ?g)))

  (:action drop
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (robot-at ?r) (carry ?b ?g))
    :effect (and (not (carry ?b ?g)) (ball-at ?b ?r) (free ?g)))
)
