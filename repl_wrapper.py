import threading
import pexpect
import json
import os
import tempfile
import re
import subprocess

class InteractiveThread(threading.Thread):
    def __init__(self, session_id, repl_path, lean_env_path, initial_context = None, timeout=600):
        super().__init__()
        self.session_id = session_id
        self.repl_path = os.path.abspath(repl_path)
        self.lean_env_path = os.path.abspath(lean_env_path)
        self.context = initial_context
        self.session = None
        
        self.cmd_response_condition = threading.Event()
        self.cmd_query_condition = threading.Event()
        self.init_complete = threading.Event()
        self.response = None

        self.stop_flag = False
        self.timer = threading.Timer(timeout, self.stop) 

    def initialize_check(self):
        try:
            if self.context == None:
                initialize_check = {"cmd": "def init_check : Nat := 42"}
                self.send_cmd(initialize_check)
            self.session.expect('"env": 0}\r\n\r\n', timeout=60)  #If context contains sorries, it will have more keys other than env
            self.init_complete.set()
        except:
            self.init_complete.set()
            print(f"Session {self.session_id}: fail to initialize lean repl")
            print(self.context)
            print(self.session.before)
            self.stop()
            # self.join()

    def send_cmd(self, cmd):
        cmd_str = json.dumps(cmd, ensure_ascii=False) 
        self.session.sendline(cmd_str + '\n')

    def submit_and_receive(self, cmd):
        if self.stop_flag: return None

        self.init_complete.wait()
        
        self.send_cmd(cmd)
        
        self.cmd_query_condition.set() 

        self.cmd_response_condition.wait()  # wait for the response
        self.cmd_response_condition.clear()
        if self.response:
            output = self.response
            self.response = None
            return output  
        return None

    def process_responses(self):
        while not self.stop_flag:
            self.cmd_query_condition.wait() #wait for input 
            self.cmd_query_condition.clear()

            if self.stop_flag:  #terminate session
                break

            try:
                self.session.expect('\r\n\r\n', timeout=30) #filter out input, pexpect print the input twice for unknown reason
                self.session.expect(['\r\n\r\n', pexpect.EOF], timeout=30)
                output = self.session.before.strip()
                output_dict = json.loads(output)

                self.response = output_dict
                self.cmd_response_condition.set()  

            except pexpect.TIMEOUT:
                print("Output timeout")  
                break # Terminate session
            except pexpect.EOF:
                print("Session ended unexpectedly.")
                break
            except json.JSONDecodeError as e:
                print(output)
                break
    
    def remove_last_comment(self):
        pattern = r'/--[^/]*?-/(\n*)$'
        self.context = re.sub(pattern, '', self.context, flags=re.DOTALL)

    def run(self):
        self.timer.start() 
        try:
            self.session = pexpect.spawn('bash', encoding='utf-8', cwd=self.lean_env_path)
            if self.context != None:
                self.remove_last_comment()
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
                    json.dump({"cmd": self.context}, temp, ensure_ascii=False)
                    temp.write("\n\n")
                    temp.flush()
                command = f'lake env {self.repl_path}/.lake/build/bin/repl < <(cat {temp.name} -)'
            else:
                command = f'lake env {self.repl_path}/.lake/build/bin/repl'
            
            self.session.sendline(command)
            self.initialize_check()
            self.process_responses()  # Continuously process responses
            self.stop()
    
        except Exception as e:
            print(f"Session {self.session_id}: An error occurred: {e}")
            self.stop()

    def stop(self):
        self.stop_flag = True
        self.init_complete.set()
        self.cmd_query_condition.set() 
        self.cmd_response_condition.set()  
        self.timer.cancel()

def evaluate_repl(*args, repl_path=os.path.join(os.getcwd(), "repl"), lean_env_path=os.path.join(os.getcwd(), "mathlib4")):
    """
    Evaluates a Lean REPL command in a separate thread.
    Args:
        *args: commands to be executed in the REPL (in a single environment)
        repl_path: path to the REPL submodule
        lean_env_path: path to the Lean project in which to execute the code
    Returns:
        A dictionary of the format {"success": bool, "errors": str}
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
        for i, cmd in enumerate(args):
            if i==0:
                json.dump({"cmd": cmd}, temp, ensure_ascii=False)
            else:
                json.dump({"cmd": cmd, "env": 0}, temp, ensure_ascii=False) #TODO: update environment number for >2 calls?
            temp.write("\n\n")
        # json.dump({"cmd": remove_last_comment(imports)}, temp, ensure_ascii=False)
        # temp.write("\n\n")
        # json.dump({"cmd": response, "env": 0}, temp, ensure_ascii=False)
        # temp.write("\n\n")
        temp.flush()
        temp_name = temp.name

    command = f'lake env {repl_path}/.lake/build/bin/repl < {temp_name}'
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=lean_env_path, timeout=60)
        result_dict = result.stdout.split("\n\n")
        outputs = []
        for res in result_dict:
            if res.strip():
                outputs.append(json.loads(res.strip()))
    except Exception as e:
        return {"success": False, "errors": e}

    if not outputs:
        return {"success": False, "errors": "No output from the REPL. If you're running this through a JetBrains IDE, you might have to run this script directly from the terminal instead."}

    if "messages" in outputs[-1]:
        errors = []
        for message in outputs[-1]["messages"]:
            if message["severity"] == 'error':
                errors.append(f"{message["data"]} (at line {message["pos"]["line"]}, column {message["pos"]["column"]})")
        if errors:
            return {"success": False, "errors": "\n".join(errors)}

    if 'env' in outputs[-1]:
        return {"success": True, "errors": None}

    return {"success": False, "errors": "No environment found."}


# TODO: this doesn't work unless the file is run through the shell. Is this just a JetBrains skill issue or is there a reason that lake can't be found?
# if __name__ == "__main__":
#     def remove_last_comment(ctx):
#         pattern = r'/--[^/]*?-/(\n*)$'
#         ctx = re.sub(pattern, '', ctx, flags=re.DOTALL)
#         return ctx
#     data = json.loads(r"""{"srcContext": "/-\nCopyright (c) 2018 Robert Y. Lewis. All rights reserved.\nReleased under Apache 2.0 license as described in the file LICENSE.\nAuthors: Robert Y. Lewis, Mario Carneiro, Johan Commelin\n-/\nimport Mathlib.NumberTheory.Padics.PadicNumbers\nimport Mathlib.RingTheory.DiscreteValuationRing.Basic\n\n/-!\n# p-adic integers\n\nThis file defines the `p`-adic integers `\u2124_[p]` as the subtype of `\u211a_[p]` with norm `\u2264 1`.\nWe show that `\u2124_[p]`\n* is complete,\n* is nonarchimedean,\n* is a normed ring,\n* is a local ring, and\n* is a discrete valuation ring.\n\nThe relation between `\u2124_[p]` and `ZMod p` is established in another file.\n\n## Important definitions\n\n* `PadicInt` : the type of `p`-adic integers\n\n## Notation\n\nWe introduce the notation `\u2124_[p]` for the `p`-adic integers.\n\n## Implementation notes\n\nMuch, but not all, of this file assumes that `p` is prime. This assumption is inferred automatically\nby taking `[Fact p.Prime]` as a type class argument.\n\nCoercions into `\u2124_[p]` are set up to work with the `norm_cast` tactic.\n\n## References\n\n* [F. Q. Gouv\u00eaa, *p-adic numbers*][gouvea1997]\n* [R. Y. Lewis, *A formal proof of Hensel's lemma over the p-adic integers*][lewis2019]\n* <https://en.wikipedia.org/wiki/P-adic_number>\n\n## Tags\n\np-adic, p adic, padic, p-adic integer\n-/\n\n\nopen Padic Metric IsLocalRing\n\nnoncomputable section\n\nvariable (p : \u2115) [hp : Fact p.Prime]\n\n/-- The `p`-adic integers `\u2124_[p]` are the `p`-adic numbers with norm `\u2264 1`. -/\ndef PadicInt : Type := {x : \u211a_[p] // \u2016x\u2016 \u2264 1}\n\n/-- The ring of `p`-adic integers. -/\nnotation \"\u2124_[\" p \"]\" => PadicInt p\n\nnamespace PadicInt\nvariable {p} {x y : \u2124_[p]}\n\n/-! ### Ring structure and coercion to `\u211a_[p]` -/\n\ninstance : Coe \u2124_[p] \u211a_[p] :=\n  \u27e8Subtype.val\u27e9\n\ntheorem ext {x y : \u2124_[p]} : (x : \u211a_[p]) = y \u2192 x = y :=\n  Subtype.ext\n\nvariable (p)\n\n/-- The `p`-adic integers as a subring of `\u211a_[p]`. -/\ndef subring : Subring \u211a_[p] where\n  carrier := { x : \u211a_[p] | \u2016x\u2016 \u2264 1 }\n  zero_mem' := by norm_num\n  one_mem' := by norm_num\n  add_mem' hx hy := (padicNormE.nonarchimedean _ _).trans <| max_le_iff.2 \u27e8hx, hy\u27e9\n  mul_mem' hx hy := (padicNormE.mul _ _).trans_le <| mul_le_one\u2080 hx (norm_nonneg _) hy\n  neg_mem' hx := (norm_neg _).trans_le hx\n\n@[simp]\ntheorem mem_subring_iff {x : \u211a_[p]} : x \u2208 subring p \u2194 \u2016x\u2016 \u2264 1 := Iff.rfl\n\nvariable {p}\n\n/-- Addition on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : Add \u2124_[p] := (by infer_instance : Add (subring p))\n\n/-- Multiplication on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : Mul \u2124_[p] := (by infer_instance : Mul (subring p))\n\n/-- Negation on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : Neg \u2124_[p] := (by infer_instance : Neg (subring p))\n\n/-- Subtraction on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : Sub \u2124_[p] := (by infer_instance : Sub (subring p))\n\n/-- Zero on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : Zero \u2124_[p] := (by infer_instance : Zero (subring p))\n\ninstance : Inhabited \u2124_[p] := \u27e80\u27e9\n\n/-- One on `\u2124_[p]` is inherited from `\u211a_[p]`. -/\ninstance : One \u2124_[p] := \u27e8\u27e81, by norm_num\u27e9\u27e9\n\n@[simp]\ntheorem mk_zero {h} : (\u27e80, h\u27e9 : \u2124_[p]) = (0 : \u2124_[p]) := rfl\n\n@[simp, norm_cast]\ntheorem coe_add (z1 z2 : \u2124_[p]) : ((z1 + z2 : \u2124_[p]) : \u211a_[p]) = z1 + z2 := rfl\n\n@[simp, norm_cast]\ntheorem coe_mul (z1 z2 : \u2124_[p]) : ((z1 * z2 : \u2124_[p]) : \u211a_[p]) = z1 * z2 := rfl\n\n@[simp, norm_cast]\ntheorem coe_neg (z1 : \u2124_[p]) : ((-z1 : \u2124_[p]) : \u211a_[p]) = -z1 := rfl\n\n@[simp, norm_cast]\ntheorem coe_sub (z1 z2 : \u2124_[p]) : ((z1 - z2 : \u2124_[p]) : \u211a_[p]) = z1 - z2 := rfl\n\n@[simp, norm_cast]\ntheorem coe_one : ((1 : \u2124_[p]) : \u211a_[p]) = 1 := rfl\n\n@[simp, norm_cast]\ntheorem coe_zero : ((0 : \u2124_[p]) : \u211a_[p]) = 0 := rfl\n\n@[simp] lemma coe_eq_zero : (x : \u211a_[p]) = 0 \u2194 x = 0 := by rw [\u2190 coe_zero, Subtype.coe_inj]\n\nlemma coe_ne_zero : (x : \u211a_[p]) \u2260 0 \u2194 x \u2260 0 := coe_eq_zero.not\n\ninstance : AddCommGroup \u2124_[p] := (by infer_instance : AddCommGroup (subring p))\n\ninstance instCommRing : CommRing \u2124_[p] := (by infer_instance : CommRing (subring p))\n\n@[simp, norm_cast]\ntheorem coe_natCast (n : \u2115) : ((n : \u2124_[p]) : \u211a_[p]) = n := rfl\n\n@[simp, norm_cast]\ntheorem coe_intCast (z : \u2124) : ((z : \u2124_[p]) : \u211a_[p]) = z := rfl\n\n/-- The coercion from `\u2124_[p]` to `\u211a_[p]` as a ring homomorphism. -/\ndef Coe.ringHom : \u2124_[p] \u2192+* \u211a_[p] := (subring p).subtype\n\n@[simp, norm_cast]\ntheorem coe_pow (x : \u2124_[p]) (n : \u2115) : (\u2191(x ^ n) : \u211a_[p]) = (\u2191x : \u211a_[p]) ^ n := rfl\n\ntheorem mk_coe (k : \u2124_[p]) : (\u27e8k, k.2\u27e9 : \u2124_[p]) = k := by simp\n\n/-- The inverse of a `p`-adic integer with norm equal to `1` is also a `p`-adic integer.\nOtherwise, the inverse is defined to be `0`. -/\ndef inv : \u2124_[p] \u2192 \u2124_[p]\n  | \u27e8k, _\u27e9 => if h : \u2016k\u2016 = 1 then \u27e8k\u207b\u00b9, by simp [h]\u27e9 else 0\n\ninstance : CharZero \u2124_[p] where\n  cast_injective m n h :=\n    Nat.cast_injective (R := \u211a_[p]) (by rw [Subtype.ext_iff] at h; norm_cast at h)\n\n@[norm_cast]\ntheorem intCast_eq (z1 z2 : \u2124) : (z1 : \u2124_[p]) = z2 \u2194 z1 = z2 := by simp\n\n/-- A sequence of integers that is Cauchy with respect to the `p`-adic norm converges to a `p`-adic\ninteger. -/\ndef ofIntSeq (seq : \u2115 \u2192 \u2124) (h : IsCauSeq (padicNorm p) fun n => seq n) : \u2124_[p] :=\n  \u27e8\u27e6\u27e8_, h\u27e9\u27e7,\n    show \u2191(PadicSeq.norm _) \u2264 (1 : \u211d) by\n      rw [PadicSeq.norm]\n      split_ifs with hne <;> norm_cast\n      apply padicNorm.of_int\u27e9\n\n/-! ### Instances\n\nWe now show that `\u2124_[p]` is a\n* complete metric space\n* normed ring\n* integral domain\n-/\n\nvariable (p)\n\ninstance : MetricSpace \u2124_[p] := Subtype.metricSpace\n\ninstance : IsUltrametricDist \u2124_[p] := IsUltrametricDist.subtype _\n\ninstance completeSpace : CompleteSpace \u2124_[p] :=\n  have : IsClosed { x : \u211a_[p] | \u2016x\u2016 \u2264 1 } := isClosed_le continuous_norm continuous_const\n  this.completeSpace_coe\n\ninstance : Norm \u2124_[p] := \u27e8fun z => \u2016(z : \u211a_[p])\u2016\u27e9\n\nvariable {p}\n\ntheorem norm_def {z : \u2124_[p]} : \u2016z\u2016 = \u2016(z : \u211a_[p])\u2016 := rfl\n\nvariable (p)\n\ninstance : NormedCommRing \u2124_[p] :=\n  { PadicInt.instCommRing with\n    dist_eq := fun \u27e8_, _\u27e9 \u27e8_, _\u27e9 => rfl\n    norm_mul := by simp [norm_def]\n    norm := norm }\n\ninstance : NormOneClass \u2124_[p] :=\n  \u27e8norm_def.trans norm_one\u27e9\n\ninstance isAbsoluteValue : IsAbsoluteValue fun z : \u2124_[p] => \u2016z\u2016 where\n  abv_nonneg' := norm_nonneg\n  abv_eq_zero' := by simp [norm_eq_zero]\n  abv_add' := fun \u27e8_, _\u27e9 \u27e8_, _\u27e9 => norm_add_le _ _\n  abv_mul' _ _ := by simp only [norm_def, padicNormE.mul, PadicInt.coe_mul]\n\nvariable {p}\n\ninstance : IsDomain \u2124_[p] := Function.Injective.isDomain (subring p).subtype Subtype.coe_injective\n\n/-! ### Norm -/\n\ntheorem norm_le_one (z : \u2124_[p]) : \u2016z\u2016 \u2264 1 := z.2\n\n@[simp]\ntheorem norm_mul (z1 z2 : \u2124_[p]) : \u2016z1 * z2\u2016 = \u2016z1\u2016 * \u2016z2\u2016 := by simp [norm_def]\n\n@[simp]\ntheorem norm_pow (z : \u2124_[p]) : \u2200 n : \u2115, \u2016z ^ n\u2016 = \u2016z\u2016 ^ n\n  | 0 => by simp\n  | k + 1 => by\n    rw [pow_succ, pow_succ, norm_mul]\n    congr\n    apply norm_pow\n\ntheorem nonarchimedean (q r : \u2124_[p]) : \u2016q + r\u2016 \u2264 max \u2016q\u2016 \u2016r\u2016 := padicNormE.nonarchimedean _ _\n\ntheorem norm_add_eq_max_of_ne {q r : \u2124_[p]} : \u2016q\u2016 \u2260 \u2016r\u2016 \u2192 \u2016q + r\u2016 = max \u2016q\u2016 \u2016r\u2016 :=\n  padicNormE.add_eq_max_of_ne\n\ntheorem norm_eq_of_norm_add_lt_right {z1 z2 : \u2124_[p]} (h : \u2016z1 + z2\u2016 < \u2016z2\u2016) : \u2016z1\u2016 = \u2016z2\u2016 :=\n  by_contra fun hne =>\n    not_lt_of_ge (by rw [norm_add_eq_max_of_ne hne]; apply le_max_right) h\n\ntheorem norm_eq_of_norm_add_lt_left {z1 z2 : \u2124_[p]} (h : \u2016z1 + z2\u2016 < \u2016z1\u2016) : \u2016z1\u2016 = \u2016z2\u2016 :=\n  by_contra fun hne =>\n    not_lt_of_ge (by rw [norm_add_eq_max_of_ne hne]; apply le_max_left) h\n\n@[simp]\ntheorem padic_norm_e_of_padicInt (z : \u2124_[p]) : \u2016(z : \u211a_[p])\u2016 = \u2016z\u2016 := by simp [norm_def]\n\ntheorem norm_intCast_eq_padic_norm (z : \u2124) : \u2016(z : \u2124_[p])\u2016 = \u2016(z : \u211a_[p])\u2016 := by simp [norm_def]\n\n@[simp]\ntheorem norm_eq_padic_norm {q : \u211a_[p]} (hq : \u2016q\u2016 \u2264 1) : @norm \u2124_[p] _ \u27e8q, hq\u27e9 = \u2016q\u2016 := rfl\n\n@[simp]\ntheorem norm_p : \u2016(p : \u2124_[p])\u2016 = (p : \u211d)\u207b\u00b9 := padicNormE.norm_p\n\ntheorem norm_p_pow (n : \u2115) : \u2016(p : \u2124_[p]) ^ n\u2016 = (p : \u211d) ^ (-n : \u2124) := by simp\n\nprivate def cauSeq_to_rat_cauSeq (f : CauSeq \u2124_[p] norm) : CauSeq \u211a_[p] fun a => \u2016a\u2016 :=\n  \u27e8fun n => f n, fun _ h\u03b5 => by simpa [norm, norm_def] using f.cauchy h\u03b5\u27e9\n\nvariable (p)\n\ninstance complete : CauSeq.IsComplete \u2124_[p] norm :=\n  \u27e8fun f =>\n    have hqn : \u2016CauSeq.lim (cauSeq_to_rat_cauSeq f)\u2016 \u2264 1 :=\n      padicNormE_lim_le zero_lt_one fun _ => norm_le_one _\n    \u27e8\u27e8_, hqn\u27e9, fun \u03b5 => by\n      simpa [norm, norm_def] using CauSeq.equiv_lim (cauSeq_to_rat_cauSeq f) \u03b5\u27e9\u27e9\n\ntheorem exists_pow_neg_lt {\u03b5 : \u211d} (h\u03b5 : 0 < \u03b5) : \u2203 k : \u2115, (p : \u211d) ^ (-(k : \u2124)) < \u03b5 := by\n  obtain \u27e8k, hk\u27e9 := exists_nat_gt \u03b5\u207b\u00b9\n  use k\n  rw [\u2190 inv_lt_inv\u2080 h\u03b5 (zpow_pos _ _)]\n  \u00b7 rw [zpow_neg, inv_inv, zpow_natCast]\n    apply lt_of_lt_of_le hk\n    norm_cast\n    apply le_of_lt\n    convert Nat.lt_pow_self _ using 1\n    exact hp.1.one_lt\n  \u00b7 exact mod_cast hp.1.pos\n\ntheorem exists_pow_neg_lt_rat {\u03b5 : \u211a} (h\u03b5 : 0 < \u03b5) : \u2203 k : \u2115, (p : \u211a) ^ (-(k : \u2124)) < \u03b5 := by\n  obtain \u27e8k, hk\u27e9 := @exists_pow_neg_lt p _ \u03b5 (mod_cast h\u03b5)\n  use k\n  rw [show (p : \u211d) = (p : \u211a) by simp] at hk\n  exact mod_cast hk\n\nvariable {p}\n\ntheorem norm_int_lt_one_iff_dvd (k : \u2124) : \u2016(k : \u2124_[p])\u2016 < 1 \u2194 (p : \u2124) \u2223 k :=\n  suffices \u2016(k : \u211a_[p])\u2016 < 1 \u2194 \u2191p \u2223 k by rwa [norm_intCast_eq_padic_norm]\n  padicNormE.norm_int_lt_one_iff_dvd k\n\ntheorem norm_int_le_pow_iff_dvd {k : \u2124} {n : \u2115} :\n    \u2016(k : \u2124_[p])\u2016 \u2264 (p : \u211d) ^ (-n : \u2124) \u2194 (p ^ n : \u2124) \u2223 k :=\n  suffices \u2016(k : \u211a_[p])\u2016 \u2264 (p : \u211d) ^ (-n : \u2124) \u2194 (p ^ n : \u2124) \u2223 k by\n    simpa [norm_intCast_eq_padic_norm]\n  padicNormE.norm_int_le_pow_iff_dvd _ _\n\n/-! ### Valuation on `\u2124_[p]` -/\n\n", "theoremStatement": "lemma valuation_coe_nonneg : 0 \u2264 (x : \u211a_[p]).valuation ", "theoremName": "PadicInt.valuation_coe_nonneg", "fileCreated": {"commit": "63d78da6a358d613bff5f34c62ed9481dbbc52c2", "date": "2023-05-25"}, "theoremCreated": {"commit": "1c080cc45939aaef448022267a6965a19e53c7bf", "date": "2024-12-21"}, "file": "mathlib/Mathlib/NumberTheory/Padics/PadicIntegers.lean", "module": "Mathlib.NumberTheory.Padics.PadicIntegers", "jsonFile": "Mathlib.NumberTheory.Padics.PadicIntegers.jsonl", "positionMetadata": {"lineInFile": 302, "tokenPositionInFile": 9232, "theoremPositionInFile": 37}, "dependencyMetadata": {"inFilePremises": true, "numInFilePremises": 3, "repositoryPremises": true, "numRepositoryPremises": 78, "numPremises": 111}, "proofMetadata": {"hasProof": true, "proof": ":= by\n  obtain rfl | hx := eq_or_ne x 0\n  \u00b7 simp\n  have := x.2\n  rwa [Padic.norm_eq_zpow_neg_valuation <| coe_ne_zero.2 hx, zpow_le_one_iff_right\u2080, neg_nonpos]\n    at this\n  exact mod_cast hp.out.one_lt", "proofType": "tactic", "proofLengthLines": 6, "proofLengthTokens": 202}}""")
#     print(evaluate_repl(
#             remove_last_comment(data["srcContext"]),
#             data["theoremStatement"] + data["proofMetadata"]["proof"]
#         )
#     )