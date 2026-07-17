# Install jq Without SSL Verification

`jq` is a command-line JSON processor. In `scripts/jiqun/hfd.sh`, it is used to parse Hugging Face metadata JSON, including gated repository checks and download file list generation.

`jq` is a system tool. It is not downloaded into the model directory.

## Ubuntu/Debian

```bash
sudo apt-get update \
  -o Acquire::https::Verify-Peer=false \
  -o Acquire::https::Verify-Host=false

sudo apt-get install -y jq \
  -o Acquire::https::Verify-Peer=false \
  -o Acquire::https::Verify-Host=false
```

The installed binary is usually:

```bash
/usr/bin/jq
```

Confirm the installation:

```bash
which jq
jq --version
```

## CentOS/RHEL/Amazon Linux

```bash
sudo yum install -y jq --setopt=sslverify=false
```

The installed binary is usually:

```bash
/usr/bin/jq
```

Confirm the installation:

```bash
which jq
jq --version
```

## Without sudo

If `sudo` is unavailable, download the standalone binary into the user directory:

```bash
mkdir -p "$HOME/bin"

curl -k -L \
  -o "$HOME/bin/jq" \
  https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64

chmod +x "$HOME/bin/jq"
export PATH="$HOME/bin:$PATH"
```

This downloads `jq` to:

```bash
$HOME/bin/jq
```

Confirm the installation:

```bash
which jq
jq --version
```

To make it permanent, add this line to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/bin:$PATH"
```
