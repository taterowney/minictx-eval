# Wrapper for the Lean REPL for proof verification and evaluation of Lean code
# Written by Jiewen Hu and Tate Rowney (CMU L3 lab)

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
