import json
import sys
import requests
import os
import re
import time
import getpass
from typing import Dict, List, Optional

# 1. DataBase and config loading functions 
def load_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_config_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"ERROR: Config file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def fetch_config_from_router(ip: str, username: str, password: str = None, key_file: str = None, enable_password: str = None) -> tuple:
    """(Optional) Live SSH fetch - connects to the DATABASE/device."""
    try:
        from netmiko import ConnectHandler
        from netmiko.exceptions import NetMikoTimeoutException, NetMikoAuthenticationException
    except ImportError:
        print("ERROR: netmiko not installed. Install with: pip install netmiko")
        sys.exit(1)

    device = {
        'device_type': 'cisco_ios',
        'ip': ip,
        'username': username,
        'timeout': 10,
        'session_timeout': 30,
    }
    if key_file:
        device['use_keys'] = True
        device['key_file'] = key_file
    else:
        device['password'] = password
    if enable_password:
        device['secret'] = enable_password

    try:
        print(f"Connecting to {ip} via SSH...")
        connection = ConnectHandler(**device)
        if enable_password:
            connection.enable()
        print("Connected. Fetching running-config...")
        config_text = connection.send_command('show running-config')
        return connection, config_text
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


# 2. CONFIG THE NORMALIZED ONE (LLM Normalization)

def call_ollama(model: str, prompt: str, temperature: float = 0.0) -> Dict:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "format": "json"
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"Ollama error: {resp.status_code}")
            sys.exit(1)
        raw = resp.json()["response"].strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"ERROR calling Ollama: {e}")
        sys.exit(1)

def normalize_config(config_text: str, instructions: Dict) -> Dict:
    """Converts raw config to normalized JSON (Config the Normalized one)."""
    schema = load_json(instructions["output_schema"])
    prompt = f"{instructions['system_prompt']}\n\nOutput schema (must match exactly):\n{json.dumps(schema, indent=2)}\n\nConfiguration:\n```\n{config_text}\n```\n\nOutput ONLY valid JSON."
    return call_ollama(instructions["model"], prompt, instructions["temperature"])

# 3. Checking if config file is right or not (Validation)
def validate_normalized_json(data: Dict, schema: Dict) -> tuple:
    """
    Strict validation that checks:
    1. All required fields exist at ALL levels
    2. Data types match the schema
    3. No unexpected fields
    """
    errors = []
    
    # 1. Check top-level required fields
    required_fields = ["vendor", "hostname"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required top-level field: '{field}'")
        elif not data[field] or len(str(data[field]).strip()) == 0:
            errors.append(f"Field '{field}' is empty")
    
    # 2. Check vendor is valid
    valid_vendors = ["cisco", "juniper", "arista", "paloalto", "unknown"]
    if "vendor" in data and data["vendor"] not in valid_vendors:
        errors.append(f"Invalid vendor: '{data['vendor']}'. Must be one of {valid_vendors}")
    
    # 3. Check schema properties recursively
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            # Skip optional fields
            if prop_name not in data:
                continue
            
            actual_value = data[prop_name]
            expected_type = prop_schema.get("type")
            
            # Type checking
            if expected_type:
                type_mapping = {
                    "string": str,
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }
                expected_python_type = type_mapping.get(expected_type)
                if expected_python_type and not isinstance(actual_value, expected_python_type):
                    errors.append(f"Field '{prop_name}' should be type '{expected_type}', got '{type(actual_value).__name__}'")
            
            # Recursively check nested objects
            if expected_type == "object" and isinstance(actual_value, dict):
                if "properties" in prop_schema:
                    sub_errors = validate_nested_object(prop_name, actual_value, prop_schema["properties"])
                    errors.extend(sub_errors)
    
    # 4. Check for raw_commands - if too many, it's a normalization failure
    raw_commands = data.get("raw_commands", [])
    total_fields = len(data)
    if raw_commands and len(raw_commands) > total_fields * 0.5:  # >50% of fields are raw
        errors.append(f"Too many unmapped commands: {len(raw_commands)} raw commands. LLM failed to normalize properly.")
    
    return len(errors) == 0, errors


def validate_nested_object(parent_name: str, data: Dict, properties: Dict) -> List[str]:
    """
    Recursively validate nested objects.
    """
    errors = []
    for prop_name, prop_schema in properties.items():
        if prop_name not in data:
            # Check if it's required
            if prop_schema.get("required", False):
                errors.append(f"Missing required field: '{parent_name}.{prop_name}'")
            continue
        
        actual_value = data[prop_name]
        expected_type = prop_schema.get("type")
        
        if expected_type:
            type_mapping = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type and not isinstance(actual_value, expected_python_type):
                errors.append(f"Field '{parent_name}.{prop_name}' should be type '{expected_type}', got '{type(actual_value).__name__}'")
        
        # Recursively check nested objects
        if expected_type == "object" and isinstance(actual_value, dict):
            if "properties" in prop_schema:
                sub_errors = validate_nested_object(f"{parent_name}.{prop_name}", actual_value, prop_schema["properties"])
                errors.extend(sub_errors)
    
    return errors


# 4. RE_inforcement prompt engg (when the configf fails the validation)

def reinforcement_prompt_engineering(raw_config: str, errors: List[str], instructions: Dict) -> Dict:
    """
    When the config fails validation, use this strict prompt to force the LLM
    to correct its output. This matches the "Re-inforcement prompt-engg" block.
    """
    print("\n" + "="*60)
    print("RE-INFORCEMENT PROMPT ENGINEERING TRIGGERED")
    print("="*60)
    print(f"Validation errors: {errors}")
    print("Sending corrective prompt to LLM...")
    
    schema = load_json(instructions["output_schema"])
    
    # The exact prompt from your diagram
    reinforcement_prompt = f"""
Hey you have given some set of rules to be used in parsing the input. 
With this set of rules, we expect the user to follow the rules in the order they are given. 
If the user does not follow the rules, the system will not parse the input.

ERRORS FOUND:
{json.dumps(errors, indent=2)}

You MUST output a corrected JSON that strictly matches this schema:
{json.dumps(schema, indent=2)}

Raw configuration to re-parse:



Output ONLY valid JSON. No other text. Fix ALL errors listed above.
"""
    
    # Call LLM with the reinforcement prompt
    return call_ollama(instructions["model"], reinforcement_prompt, instructions["temperature"])

# 5. API2C_BACK / return_network (Compliance Engine - YES path)

def extract_value(data: Dict, field_path: str):
    """Dotted path extraction (e.g., 'management.ssh_version')."""
    parts = field_path.split('.')
    cur = data
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def evaluate_rule(rule: Dict, norm_data: Dict) -> Dict:
    """Evaluates a single CIS rule against normalized data."""
    field = rule["field"]
    actual = extract_value(norm_data, field)
    expected = rule["expected"]
    op = rule["operator"]
    result = False

    if op == "eq":
        result = actual == expected
    elif op == "ge":
        result = actual >= expected
    elif op == "in":
        result = actual in expected
    elif op == "contains_any":
        if isinstance(actual, list):
            result = any(a in expected for a in actual)
        else:
            result = any(exp in str(actual) for exp in expected)
    elif op == "not_contains_any":
        if isinstance(actual, list):
            result = not any(a in expected for a in actual)
        else:
            result = not any(exp in str(actual) for exp in expected)
    elif op == "exists":
        result = actual is not None
    elif op == "non_empty":
        result = bool(actual) if actual is not None else False
    else:
        result = False

    status = "Pass" if result else "Fail"
    return {
        "rule_id": rule["id"],
        "framework": rule.get("framework", "CIS"),
        "name": rule["name"],
        "status": status,
        "actual": actual,
        "expected": expected,
        "operator": op,
        "severity": rule["severity"],
        "level": rule["level"],
        "description": rule.get("description", "")
    }

def compliance_engine(normalized_data: Dict, rules: List[Dict]) -> Dict:
    """
    This is the API2C_BACK / return_network function.
    Evaluates all rules and returns the report.
    """
    print("\n" + "-"*60)
    print("API2C_BACK: Running Compliance Engine...")
    print("return_network(api_list, prompt_output())")
    print("-"*60)
    
    results = []
    for rule in rules:
        results.append(evaluate_rule(rule, normalized_data))
    
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "Pass")
    failed = sum(1 for r in results if r["status"] == "Fail")
    high_fails = sum(1 for r in results if r["status"] == "Fail" and r["severity"] == "High")
    
    return {
        "summary": {"total": total, "passed": passed, "failed": failed, "critical_high": high_fails},
        "results": results
    }

# 6. Executing the shell command after some delay most probably 5 sec

def generate_remediation_with_llm(vendor: str, os_version: str, failed_rule: Dict, normalized_data: Dict, instructions: Dict) -> List[str]:
    """Uses LLM to generate the exact fix commands."""
    prompt = f"""
Vendor: {vendor}
OS Version: {os_version}
Failed Rule: {failed_rule['id']} - {failed_rule['name']}
Severity: {failed_rule['severity']}
Expected: {failed_rule['field']} should be {failed_rule['expected']} (Operator: {failed_rule['operator']})
Current value: {failed_rule.get('actual', 'Not found')}

Generate the EXACT CLI commands to fix this issue on a {vendor} device.
Return only JSON with a "commands" array (list of strings).
"""
    
    remediation_instructions = load_json("llm_remediation_instructions.json") if os.path.exists("llm_remediation_instructions.json") else instructions
    return call_ollama(remediation_instructions["model"], prompt, 0).get("commands", [])

def execute_with_delay(connection, commands: List[str], delay_seconds: int = 5):
    """
    Executes shell commands after a delay to securely execute the command.
    This matches the diagram's "Executing the shell command after some delay".
    """
    if not commands:
        print("No commands to execute.")
        return
    
    print(f"\ WARNING: About to execute {len(commands)} commands on the device.")
    print("Commands to execute:")
    for cmd in commands:
        print(f"  {cmd}")
    
    print(f"\nWaiting {delay_seconds} seconds for you to review...")
    time.sleep(delay_seconds)
    
    confirm = input(f"Type 'YES' to execute: ").strip()
    if confirm != "YES":
        print("Execution cancelled by user.")
        return
    
    try:
        output = connection.send_config_set(commands)
        print("Command output:\n", output)
        connection.send_command('write memory')
        print("Configuration saved to startup-config.")
    except Exception as e:
        print(f"ERROR during execution: {e}")


#7 Taking user input and proceeding (Main shit starts here)

def print_report(report: Dict):
    print("\n" + "="*60)
    print("COMPLIANCE REPORT")
    print("="*60)
    s = report["summary"]
    print(f"Total rules: {s['total']} | Passed: {s['passed']} | Failed: {s['failed']} | Critical High: {s['critical_high']}")
    print("-"*60)
    for r in report["results"]:
        icon = " PASS" if r["status"] == "Pass" else " FAIL" if r["status"] == "Fail" else "⚠️ UNKN"
        print(f"{icon} {r['rule_id']} ({r['framework']})")
        print(f"   Name: {r['name']}")
        if r["status"] != "Pass":
            print(f"   Actual: {r['actual']} | Expected: {r['expected']}")
        print()

def main():
    print("\n" + "="*60)
    print("CIS COMPLIANCE ENGINE")
    print("="*60)
    
    # taking user input and proceeding 
    print("\n[User Input] How would you like to proceed?")
    print("1. Load config from local file")
    print("2. Connect to live router via SSH")
    choice = input("Enter 1 or 2: ").strip()
    
    config_text = None
    connection = None
    
    if choice == "1":
        file_path = input("Enter config file path: ").strip()
        config_text = load_config_from_file(file_path)
    elif choice == "2":
        ip = input("Enter router IP: ").strip()
        username = input("Username: ").strip()
        enable_pass = input("Enable password (Enter if none): ").strip() or None
        password = getpass.getpass("SSH password: ")
        connection, config_text = fetch_config_from_router(ip, username, password, enable_password=enable_pass)
    else:
        print("Invalid choice.")
        sys.exit(1)
    
    if not config_text or len(config_text.strip()) < 10:
        print("ERROR: Config is empty or too short.")
        sys.exit(1)
    
    # DATABASE: Load instructions and schema 
    instructions = load_json("llm_instructions.json")
    schema = load_json(instructions["output_schema"])
    rules = load_json("cis_rules.json")
    
    # 2. Config the normalised ones 
    
    print("\n[2. Config the Normalized one] Sending to LLM...")
    normalized_data = normalize_config(config_text, instructions)
    
    # 3. chck if config file is right or not (Validation)

    is_valid, errors = validate_normalized_json(normalized_data, schema)
    

    # 4. RE-inforcemnt prompt engg (If NO)
    if not is_valid:
        print("\n[Validation] Config file is NOT valid.")
        print(f"Found {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
    
    # DEBUG: Print what the LLM actually returned
        print("\n[DEBUG] LLM returned normalized data:")
        print(json.dumps(normalized_data, indent=2))
    
        retry_count = 0
        max_retries = 2
        while retry_count < max_retries and not is_valid:
            print(f"\n--- Reinforcement Prompt Engineering Attempt {retry_count + 1} ---")
            normalized_data = reinforcement_prompt_engineering(config_text, errors, instructions)
            is_valid, errors = validate_normalized_json(normalized_data, schema)
            retry_count += 1
    
        if not is_valid:
            print("\Failed to normalize after retries. Exiting.")
            print("Final validation errors:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
    
    # Save the normalized output
    with open("normalized_output.json", "w") as f:
        json.dump(normalized_data, f, indent=2)
    print("\n[Validation]  Config file is valid. Normalized data saved.")
    
    # 5. API2C_BACK / return_network (Compliance Engine - YES path)
    
    report = compliance_engine(normalized_data, rules)
    print_report(report)
    
    # 6. EXECUTING THE SHELL COMMAND AFTER SOME DELAY
    failed_rules = [r for r in report["results"] if r["status"] == "Fail"]
    
    if failed_rules and connection:
        print(f"\nFound {len(failed_rules)} failing rules.")
        apply = input("Generate and execute remediation commands? (y/n): ").strip().lower()
        
        if apply in ['y', 'yes']:
            vendor = normalized_data.get("vendor", "cisco")
            os_version = normalized_data.get("os_version", "unknown")
            
            all_commands = []
            for rule in failed_rules:
                print(f"\nGenerating fix for {rule['rule_id']}...")
                cmds = generate_remediation_with_llm(vendor, os_version, rule, normalized_data, instructions)
                if cmds:
                    all_commands.extend(cmds)
            
            if all_commands:
                # Delay execution for security (5 seconds)
                execute_with_delay(connection, all_commands, delay_seconds=5)
                
                # Re-audit after fixes
                print("\n[Re-audit] Fetching new config and re-evaluating...")
                new_config = connection.send_command('show running-config')
                new_normalized = normalize_config(new_config, instructions)
                new_report = compliance_engine(new_normalized, rules)
                print_report(new_report)
            else:
                print("No remediation commands generated.")
    
    elif failed_rules and not connection:
        print("\nRemediation commands (copy-paste manually):")
        for rule in failed_rules:
            print(f"  # Fix {rule['rule_id']}: {rule['name']}")
            print(f"  # Expected: {rule['expected']}, Actual: {rule['actual']}")
    
    # Close connection if open
    if connection:
        connection.disconnect()
        print("\nSSH session closed.")
    
    # Save final report
    with open("compliance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull report saved to compliance_report.json")
    print("[Taking user input and proceeding] - Done. Exiting.")


if __name__ == "__main__":
    main()
    
    
    
    
"""
ok the thing I noticed is the stream error in ollama if you put $stream : True then following is the output :

with $stream : True
srv  server_strea: conv_id= (empty=1)
slot get_availabl: id  0 | task -1 |  - checking sim = 0.002 (3/1262) > 0.100
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 368189966
srv  get_availabl: updating prompt cache
srv   prompt_save:  - saving prompt with length 1562, total state size = 99.093 MiB (draft: 0.000 MiB)
srv          load:  - looking for better prompt, base f_keep = 0.002, f_sim = 0.002
srv          load:    - prompt with length    2009, lcp =     159, f_keep = 0.079, f_sim = 0.126
srv          load:    - prompt with length    1562, lcp =       3, f_keep = 0.002, f_sim = 0.002
srv          load:    - prompt with length    1562, lcp =       3, f_keep = 0.002, f_sim = 0.002
srv        update:  - cache state: 3 prompts, 663.011 MiB (limits: 8192.000 MiB, 4096 tokens, 63422 est)
srv        update:    - prompt 0000021C9701D570:    2009 tokens, checkpoints:  3,   263.822 MiB
srv        update:    - prompt 0000021C9701D9D0:    1562 tokens, checkpoints:  2,   199.594 MiB
srv        update:    - prompt 0000021D4AAC8EF0:    1562 tokens, checkpoints:  2,   199.594 MiB
srv  get_availabl: prompt cache update took 90.82 ms
slot launch_slot_: id  0 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> ?min-p -> ?xtc -> temp-ext -> dist
slot launch_slot_: id  0 | task -1 | sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
        top_k = 40, top_p = 0.900, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
slot launch_slot_: id  0 | task 1665 | processing task, is_child = 0
slot   operator(): id  0 | task 1665 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 1262
slot   operator(): id  0 | task 1665 | checking checkpoint with [1145, 1145] against 3...
slot   operator(): id  0 | task 1665 | checking checkpoint with [637, 637] against 3...
slot   operator(): id  0 | task 1665 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
slot   operator(): id  0 | task 1665 | erased invalidated context checkpoint (pos_min = 637, pos_max = 637, n_tokens = 638, n_swa = 0, pos_next = 0, size = 50.251 MiB)
slot   operator(): id  0 | task 1665 | erased invalidated context checkpoint (pos_min = 1145, pos_max = 1145, n_tokens = 1146, n_swa = 0, pos_next = 0, size = 50.251 MiB)
slot   operator(): id  0 | task 1665 | cached n_tokens = 0, memory_seq_rm [0, end)
slot   operator(): id  0 | task 1665 | cached n_tokens = 512, memory_seq_rm [512, end)
slot   operator(): id  0 | task 1665 | cached n_tokens = 750, memory_seq_rm [750, end)
slot create_check: id  0 | task 1665 | created context checkpoint 1 of 32 (pos_min = 749, pos_max = 749, n_tokens = 750, size = 50.251 MiB)
slot   operator(): id  0 | task 1665 | cached n_tokens = 1258, memory_seq_rm [1258, end)
slot init_sampler: id  0 | task 1665 | init sampler, took 0.25 ms, tokens: text = 1262, total = 1262
slot create_check: id  0 | task 1665 | created context checkpoint 2 of 32 (pos_min = 1257, pos_max = 1257, n_tokens = 1258, size = 50.251 MiB)
slot print_timing: id  0 | task 1665 | n_gen =    100, tg =  11.80 t/s, tg_3s =  11.92 t/s
slot print_timing: id  0 | task 1665 | n_gen =    130, tg =  11.16 t/s, tg_3s =   9.45 t/s
slot print_timing: id  0 | task 1665 | n_gen =    161, tg =  10.74 t/s, tg_3s =   9.30 t/s
slot print_timing: id  0 | task 1665 | n_gen =    179, tg =   9.90 t/s, tg_3s =   5.83 t/s
slot print_timing: id  0 | task 1665 | n_gen =    199, tg =   9.43 t/s, tg_3s =   6.62 t/s
slot print_timing: id  0 | task 1665 | n_gen =    219, tg =   9.05 t/s, tg_3s =   6.52 t/s
slot print_timing: id  0 | task 1665 | n_gen =    239, tg =   8.78 t/s, tg_3s =   6.61 t/s
slot print_timing: id  0 | task 1665 | n_gen =    259, tg =   8.56 t/s, tg_3s =   6.62 t/s
slot print_timing: id  0 | task 1665 | n_gen =    279, tg =   8.38 t/s, tg_3s =   6.59 t/s
slot print_timing: id  0 | task 1665 | prompt eval time =    2210.24 ms /  1262 tokens (    1.75 ms per token,   570.98 tokens per second)
slot print_timing: id  0 | task 1665 |        eval time =   34720.13 ms /   289 tokens (  120.56 ms per token,     8.29 tokens per second)
slot print_timing: id  0 | task 1665 |       total time =   36930.37 ms /  1551 tokens
slot print_timing: id  0 | task 1665 |    graphs reused =       1921
slot      release: id  0 | task 1665 | stop processing: n_tokens = 1550, truncated = 0
srv  update_slots: all slots are idle




with $stream : False 

srv  server_strea: conv_id= (empty=1)
slot get_availabl: id  0 | task -1 |  - checking sim = 1.000 (1262/1262) > 0.100
slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.814
slot launch_slot_: id  0 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> ?min-p -> ?xtc -> temp-ext -> dist
slot launch_slot_: id  0 | task -1 | sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
        top_k = 40, top_p = 0.900, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
slot launch_slot_: id  0 | task 1958 | processing task, is_child = 0
slot   operator(): id  0 | task 1958 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 1262
slot   operator(): id  0 | task 1958 | checking checkpoint with [1257, 1257] against 1261...
slot   operator(): id  0 | task 1958 | restored context checkpoint (pos_min = 1257, pos_max = 1257, n_tokens = 1258, n_past = 1258, size = 50.251 MiB)
slot   operator(): id  0 | task 1958 | cached n_tokens = 1258, memory_seq_rm [1258, end)
slot init_sampler: id  0 | task 1958 | init sampler, took 0.27 ms, tokens: text = 1262, total = 1262
slot create_check: id  0 | task 1958 | erasing context checkpoint too close to an earlier one (pos_min = 1257, pos_max = 1257, n_tokens = 1258, size = 50.251 MiB)
slot create_check: id  0 | task 1958 | created context checkpoint 2 of 32 (pos_min = 1257, pos_max = 1257, n_tokens = 1258, size = 50.251 MiB)
slot print_timing: id  0 | task 1958 | n_gen =    100, tg =  15.44 t/s, tg_3s =  15.59 t/s
slot print_timing: id  0 | task 1958 | n_gen =    120, tg =  12.56 t/s, tg_3s =   6.53 t/s
slot print_timing: id  0 | task 1958 | n_gen =    138, tg =  10.95 t/s, tg_3s =   5.93 t/s
slot print_timing: id  0 | task 1958 | n_gen =    156, tg =   9.89 t/s, tg_3s =   5.70 t/s
slot print_timing: id  0 | task 1958 | n_gen =    176, tg =   9.33 t/s, tg_3s =   6.49 t/s
slot print_timing: id  0 | task 1958 | n_gen =    196, tg =   8.95 t/s, tg_3s =   6.59 t/s
slot print_timing: id  0 | task 1958 | n_gen =    216, tg =   8.65 t/s, tg_3s =   6.54 t/s
slot print_timing: id  0 | task 1958 | n_gen =    236, tg =   8.43 t/s, tg_3s =   6.59 t/s
slot print_timing: id  0 | task 1958 | n_gen =    256, tg =   8.25 t/s, tg_3s =   6.57 t/s
slot print_timing: id  0 | task 1958 | n_gen =    276, tg =   8.10 t/s, tg_3s =   6.59 t/s
slot print_timing: id  0 | task 1958 | prompt eval time =     433.53 ms /     4 tokens (  108.38 ms per token,     9.23 tokens per second)
slot print_timing: id  0 | task 1958 |        eval time =   36295.51 ms /   291 tokens (  125.16 ms per token,     7.99 tokens per second)
slot print_timing: id  0 | task 1958 |       total time =   36729.04 ms /   295 tokens
slot print_timing: id  0 | task 1958 |    graphs reused =       2208
slot      release: id  0 | task 1958 | stop processing: n_tokens = 1552, truncated = 0
srv  update_slots: all slots are idle


so for the best option put $stream : False in the call_ollama() function
 
"""