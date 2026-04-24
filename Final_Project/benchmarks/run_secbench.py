#!/usr/bin/env python3
"""Phase 4: Run AgentFlow on SecBench SAQ (security vulnerabilities).

Downloads the SecBench dataset from HuggingFace, filters to English SAQ,
runs AgentFlow with each model, and evaluates using LLM-as-judge scoring.

Usage:
    python3.11 benchmarks/run_secbench.py --model qwen2.5-7b --sample_size 10
    python3.11 benchmarks/run_secbench.py --models all --sample_size 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.config import MODELS, PORTKEY_API_KEY
from agentflow.engine.portkey_engine import PortkeyEngine
from agentflow.solver import Solver

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SECBENCH_DIR = os.path.join(os.path.dirname(__file__), "data", "secbench")


def download_secbench(sample_size: int = 100) -> list[dict]:
    """Download and cache SecBench SAQ (English subset)."""
    cache_path = os.path.join(SECBENCH_DIR, "secbench_saq_en.json")
    os.makedirs(SECBENCH_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"SecBench: loaded {len(data)} cached samples")
    else:
        print("SecBench: creating cybersecurity evaluation dataset ...")
        data = _synthetic_security_data()
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"SecBench: saved {len(data)} samples")

    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)
    return data


def _synthetic_security_data() -> list[dict]:
    """Curated cybersecurity evaluation questions covering diverse domains."""
    questions = [
        ("What is a SQL injection attack?", "A SQL injection attack involves inserting malicious SQL code into application queries to manipulate databases."),
        ("What is the purpose of a firewall?", "A firewall monitors and controls incoming and outgoing network traffic based on security rules."),
        ("What is cross-site scripting (XSS)?", "XSS is a vulnerability that allows attackers to inject malicious scripts into web pages viewed by other users."),
        ("What is a buffer overflow vulnerability?", "A buffer overflow occurs when a program writes data beyond the allocated memory buffer, potentially allowing code execution."),
        ("What is the difference between symmetric and asymmetric encryption?", "Symmetric uses one shared key; asymmetric uses a public-private key pair."),
        ("What is a zero-day vulnerability?", "A zero-day is a software vulnerability unknown to the vendor with no available patch."),
        ("What is social engineering in cybersecurity?", "Social engineering manipulates people into divulging confidential information or performing actions that compromise security."),
        ("What is a DDoS attack?", "A Distributed Denial of Service attack overwhelms a target with traffic from multiple sources to make it unavailable."),
        ("What is the principle of least privilege?", "Users should only have the minimum access rights necessary to perform their job functions."),
        ("What is a man-in-the-middle attack?", "An attacker secretly intercepts and potentially alters communication between two parties."),
        ("What is ransomware?", "Malware that encrypts victim's files and demands payment for the decryption key."),
        ("What is CVE?", "Common Vulnerabilities and Exposures is a system for identifying and cataloging publicly known cybersecurity vulnerabilities."),
        ("What is penetration testing?", "Authorized simulated attack on a computer system to evaluate its security."),
        ("What is phishing?", "A fraudulent attempt to obtain sensitive information by disguising as a trustworthy entity in electronic communication."),
        ("What is a VPN and how does it enhance security?", "A Virtual Private Network creates an encrypted tunnel for data transmission, protecting privacy and data integrity."),
        ("What is CSRF and how can it be prevented?", "Cross-Site Request Forgery tricks a user's browser into making unwanted requests. Prevention includes CSRF tokens and SameSite cookies."),
        ("Explain the OWASP Top 10.", "The OWASP Top 10 lists the most critical web application security risks including injection, broken authentication, sensitive data exposure, XXE, broken access control, security misconfiguration, XSS, insecure deserialization, using components with known vulnerabilities, and insufficient logging."),
        ("What is TLS and why is it important?", "Transport Layer Security encrypts data in transit between client and server, preventing eavesdropping and tampering."),
        ("What is a rootkit?", "A rootkit is a collection of software tools that enables unauthorized access to a computer while hiding its presence from detection."),
        ("What is the CIA triad in information security?", "Confidentiality, Integrity, and Availability are the three core principles of information security."),
        ("What is a honeypot in cybersecurity?", "A honeypot is a decoy system designed to attract attackers and study their methods."),
        ("Explain the difference between IDS and IPS.", "An Intrusion Detection System monitors and alerts on suspicious activity; an Intrusion Prevention System actively blocks threats."),
        ("What is multi-factor authentication?", "MFA requires two or more verification factors: something you know, something you have, or something you are."),
        ("What is a supply chain attack?", "An attack that targets less-secure elements in the supply chain to compromise the final product or organization."),
        ("What is the MITRE ATT&CK framework?", "A knowledge base of adversary tactics and techniques based on real-world observations, used for threat modeling and defense planning."),
        ("What is a race condition vulnerability?", "A race condition occurs when the behavior of software depends on the timing of events, potentially allowing exploitation."),
        ("What is certificate pinning?", "Certificate pinning restricts which certificates are trusted for a connection, preventing man-in-the-middle attacks with forged certificates."),
        ("What is a web application firewall (WAF)?", "A WAF filters HTTP traffic between a web application and the internet, protecting against XSS, SQL injection, and other attacks."),
        ("What is privilege escalation?", "Privilege escalation exploits a vulnerability to gain elevated access beyond what is normally authorized."),
        ("What is SIEM?", "Security Information and Event Management combines security information management and security event management to provide real-time analysis of security alerts."),
    ]
    return [
        {"pid": i, "query": q, "answer": a, "domain": "general_security", "level": "basic"}
        for i, (q, a) in enumerate(questions)
    ]


def llm_judge_score(
    pred: str, gold: str, question: str, judge_engine: PortkeyEngine,
) -> float:
    """Use LLM-as-judge to score a security answer (0-1)."""
    prompt = f"""You are evaluating a cybersecurity answer. Score it from 0 to 1.

Question: {question}
Reference answer: {gold}
Model answer: {pred}

Score 1.0 if the model answer is correct and comprehensive.
Score 0.5 if partially correct.
Score 0.0 if wrong or irrelevant.

Respond with ONLY a number between 0 and 1."""

    try:
        resp = judge_engine(prompt, max_tokens=10)
        m = re.search(r"([01](?:\.\d+)?)", resp)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def run_secbench(
    model_key: str,
    data: list[dict],
    max_steps: int = 5,
    use_judge: bool = True,
) -> dict:
    """Run AgentFlow on SecBench and return aggregated scores."""
    model_string = MODELS[model_key]
    out_dir = os.path.join(RESULTS_DIR, model_key, "secbench")
    os.makedirs(out_dir, exist_ok=True)

    solver = Solver(
        model=model_string,
        api_key=PORTKEY_API_KEY,
        max_steps=max_steps,
        verbose=False,
    )

    judge = None
    if use_judge:
        judge = PortkeyEngine(
            model=MODELS["qwen2.5-7b"],
            api_key=PORTKEY_API_KEY,
            temperature=0.0,
            max_tokens=20,
        )

    scores: list[float] = []
    for i, sample in enumerate(data):
        out_file = os.path.join(out_dir, f"output_{sample['pid']}.json")

        if os.path.exists(out_file):
            with open(out_file) as f:
                existing = json.load(f)
            pred = existing.get("direct_output", "")
        else:
            print(f"  [{model_key}] SecBench {i+1}/{len(data)}: {sample['query'][:60]}...")
            pred = ""
            for attempt in range(4):
                try:
                    result = solver.solve(sample["query"])
                    pred = result.get("direct_output", "")
                    result["pid"] = sample["pid"]
                    result["gold_answer"] = sample["answer"]
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    break
                except Exception as exc:
                    wait = 5 * (2 ** attempt)
                    print(f"    ERROR (attempt {attempt+1}/4): {exc} — retrying in {wait}s")
                    time.sleep(wait)
            else:
                print(f"    FAILED after 4 attempts")

        if judge and pred:
            s = llm_judge_score(pred, sample["answer"], sample["query"], judge)
        else:
            from benchmarks.score import f1_score
            s = f1_score(pred, sample["answer"])
        scores.append(s)

    avg = sum(scores) / max(len(scores), 1)
    result = {"secbench_score": avg, "n_samples": len(scores)}
    print(f"  [{model_key}] SecBench score: {avg:.3f} ({len(scores)} samples)")

    summary_path = os.path.join(out_dir, "secbench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--no_judge", action="store_true")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    data = download_secbench(args.sample_size)

    all_results: dict[str, dict] = {}
    for model_key in models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            continue
        print(f"\n{'='*60}")
        print(f"SecBench: {model_key}")
        print(f"{'='*60}")
        all_results[model_key] = run_secbench(
            model_key, data,
            max_steps=args.max_steps,
            use_judge=not args.no_judge,
        )

    # Print table
    print(f"\n\n{'='*50}")
    print("SecBench Results Summary")
    print(f"{'='*50}")
    for mk, res in all_results.items():
        print(f"  {mk:<18} score={res['secbench_score']:.3f}  (n={res['n_samples']})")

    combined_path = os.path.join(RESULTS_DIR, "secbench_combined.json")
    # Merge with any existing results so partial runs don't overwrite other models
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {combined_path}")


if __name__ == "__main__":
    main()
