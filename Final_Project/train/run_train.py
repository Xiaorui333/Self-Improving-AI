"""Modal launcher for Flow-GRPO training and model serving.

Training:
    modal run train/run_train.py --smoke-test           # NQ+Math smoke
    modal run train/run_train.py                        # NQ+Math full
    modal run train/run_train.py --humaneval --smoke-test
    modal run train/run_train.py --agentflow --smoke-test   # AgentFlow Planner
    modal run train/run_train.py --agentflow                # AgentFlow Planner full

Serving (after --agentflow training):
    modal run train/run_train.py --serve                # deploy API endpoint

The timeout is 600s (10 min) for dev runs and 10800s (3h) for full runs.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "PYTHONUNBUFFERED": "1",
        # Prevent process hang on Modal's kernel 4.4 (below recommended 5.5).
        # gradient_checkpointing + old kernel = deadlock in shared-memory ops.
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
    })
    .pip_install("torch", "torchvision")
    .pip_install([
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "pillow",
        "fastapi",
        "uvicorn",
    ])
    .add_local_dir(".", "/root")
)

app = modal.App("agentflow-flow-grpo")

hf = modal.Volume.from_name("hf", create_if_missing=True)
runs = modal.Volume.from_name("runs", create_if_missing=True)

# Shared volume between AgentFlow Planner training and serving.
grpo_vol = modal.Volume.from_name("grpo-agentflow-vol", create_if_missing=True)
VOLUME_ROOT = "/vol/flow_grpo_agentflow"


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
        VOLUME_ROOT: grpo_vol,
    },
    timeout=600,
)
def train_smoke(script: str = "train/flow_grpo.py"):
    """10-minute dev run for pipeline verification."""
    import subprocess, os
    env = {**os.environ, "SMOKE_TEST": "1"}
    subprocess.run(["python", f"/root/{script}"], check=True, env=env)
    # Commit volume so serving can see the checkpoint immediately
    grpo_vol.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
        VOLUME_ROOT: grpo_vol,
    },
    timeout=10800,
)
def train_full(script: str = "train/flow_grpo.py"):
    """Full training run (up to 3 hours). Only use after smoke test passes."""
    import subprocess, os
    env = {**os.environ, "SMOKE_TEST": "0"}
    subprocess.run(["python", f"/root/{script}"], check=True, env=env)
    grpo_vol.commit()


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    script: str = "train/flow_grpo.py",
    humaneval: bool = False,
    secbench: bool = False,
    agentflow: bool = False,
    serve: bool = False,
):
    """
    Entrypoint for all Flow-GRPO training scripts and model serving.

    Training:
        modal run train/run_train.py --smoke-test                    # NQ+Math smoke test
        modal run train/run_train.py                                 # NQ+Math full run
        modal run train/run_train.py --humaneval --smoke-test        # HumanEval smoke test
        modal run train/run_train.py --humaneval                     # HumanEval full run
        modal run train/run_train.py --agentflow --smoke-test        # AgentFlow Planner smoke
        modal run train/run_train.py --agentflow                     # AgentFlow Planner full

    Serving (after --agentflow training completes):
        modal run train/run_train.py --serve

    Full pipeline for 10 benchmarks:
        1. modal run train/run_train.py --agentflow            (train)
        2. modal run train/run_train.py --serve                (deploy API)
        3. export GRPO_MODEL_URL=<url printed in step 2>/v1
        4. python3.11 benchmarks/run_benchmark.py --model qwen3.5-0.8b-grpo --benchmarks all
    """
    if serve:
        _deploy_serve()
        return

    if agentflow:
        script = "train/flow_grpo_agentflow.py"
    elif humaneval:
        script = "train/flow_grpo_humaneval.py"
    elif secbench:
        script = "train/flow_grpo_secbench.py"

    if smoke_test:
        print(f"Running SMOKE TEST (10-min timeout): {script}")
        train_smoke.remote(script)
    else:
        print(f"Running FULL TRAINING (3-hour timeout): {script}")
        train_full.remote(script)


def _deploy_serve():
    """Deploy the GRPO model serving endpoint on Modal."""
    import subprocess
    import sys

    print("Deploying GRPO model serving endpoint...")
    print("(uses checkpoint saved by --agentflow training)")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", "train/serve_grpo_model.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\nDeploy failed. Try manually:")
        print("  modal deploy train/serve_grpo_model.py")
    else:
        print("\nEndpoint deployed. Find your URL in the Modal dashboard:")
        print("  https://modal.com/apps")
        print("\nThen set the environment variable and run benchmarks:")
        print("  export GRPO_MODEL_URL=https://<your-endpoint>   # origin only, no /v1")
        print("  python3.11 benchmarks/run_benchmark.py --model qwen3.5-0.8b-grpo --benchmarks all --sample_size 20")
