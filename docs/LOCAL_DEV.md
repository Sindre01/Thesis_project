## 🌐 Local Development with Ollama API on FOX

The steps below use bash commands to set up and run Ollama on the FOX cluster, allowing you to interact with it from your local machine.
The setup assumes you have SSH access to the FOX cluster and Ollama installed on it.
Steps 1-2 is for initial setup, and steps 3-5 (and more) is automated in script `run_interactive_ollama.sh`.

### 1 · SSH config

```bash
nano ~/.ssh/config
```

Add:

```text
Host fox
    HostName fox.educloud.no
    User <username>   # e.g., ec-sindrre
    IdentityFile ~/.ssh/id_rsa
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 8h
    ServerAliveInterval 120
    ServerAliveCountMax 10
```

### 2 · Start (or reuse) an SSH master session

```bash
ssh fox   # asks for pwd & 2FA once per ControlPersist window
exit      # leave but keep master alive
```

Check/close:

```bash
ssh -O check fox   # status
ssh -O exit  fox   # stop master + sockets
```

### 3 · Run **Ollama** on a compute/GPU node

```bash
# on gpu‑10, gpu‑11, ...             (never on the login node!)
# First a gpu node must be requested though slurm (batch job script with the command bellow OR an interactive session)
ollama serve
```

Health‑check:

```bash
curl localhost:11434   # → “Ollama is running”
```

### 4 · Port‑forward the API to your laptop

```bash
ssh -f -N -L 11434:<gpu-nodename>:11434 fox
curl http://localhost:11434   # same “Ollama is running” message
```

Now every local tool (Python, VS Code, etc.) can use `http://localhost:11434` for accesing the ollama api running on the FOX cluster.

### 5 · Wrap up

When done:

```bash
ssh -O exit fox   # closes port forward & master session
```

---

### Optional – Custom model cache location

Home directory quota is tight; cache models under project storage instead:

```bash
export OLLAMA_MODELS=/cluster/work/projects/ec12/<username>/ollama-models
```

Add the line to your `~/.bashrc` on FOX and re‑login.

---

## Steps 3-5 (and more) is automated in script `run_interactive_ollama.sh`