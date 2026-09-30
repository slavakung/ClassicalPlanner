(define (problem gripper-mini-instance)
  (:domain gripper-mini)
  (:objects rooma roomb - room ball1 - ball left - gripper)
  (:init
    (robot-at rooma)
    (ball-at ball1 rooma)
    (free left))
  (:goal (ball-at ball1 roomb))
)
