
from pantograph import Server
from pantograph.expr import TacticHave, TacticCalc, TacticExpr

server = Server(project_path="test-envs/minictx-v2/mathlib4")

state = server.load_header("""import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Tactic""")

# state = server.load_sorry("""theorem Problem1_1 {a b : ℤ} (h : a ∣ b) : a ∣ (a^2 - b^2) := by sorry""")

print(state)
# state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
#
# print(state0)
#
# state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
# print(state1)