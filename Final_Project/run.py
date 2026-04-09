import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install("torch")
    .pip_install([
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "wandb",
    ])
    .add_local_dir(".", "/root")
)

app = modal.App("grpo-lora")

hf = modal.Volume.from_name("hf", create_if_missing=True)
runs = modal.Volume.from_name("runs", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
    },
    timeout=10800,
)
def run(code_path: str, smoke_test: bool = False):
    import subprocess
    env = {**__import__("os").environ, "SMOKE_TEST": "1" if smoke_test else "0"}
    subprocess.run(["python", code_path], check=True, env=env)


@app.local_entrypoint()
def main(smoke_test: bool = False):
    run.remote("/root/tinyzero.py", smoke_test=smoke_test)
