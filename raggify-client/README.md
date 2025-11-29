# Raggify Client

[![Tests](https://github.com/jun76/raggify/actions/workflows/tests.yml/badge.svg)](https://github.com/jun76/raggify/actions/workflows/tests.yml) [![codecov](https://codecov.io/github/jun76/raggify/graph/badge.svg?token=TFE0CSETR8)](https://codecov.io/github/jun76/raggify)

<img alt="Image" src="https://raw.githubusercontent.com/jun76/raggify/main/media/Raggify.png" />

**Raggify Client** is a lightweight library that extracts only the REST API client portion from **Raggify**, a comprehensive library that includes server modules.

For details on the Raggify library, please refer to [this page](https://github.com/jun76/raggify).

# 🚀 How to Install

To install, run:

```bash
pip install raggify-client
```

raggify-client requires a raggify server to be running on the backend.
You can specify server `host` and `port` in `/etc/raggify-client/config.yaml`.

```yaml
host: localhost
port: 8000
topk: 20
log_level: DEBUG
```

The default config.yaml is generated when raggify-client is run for the first time.

```bash
raggify-cilent --help
```

# 📚 Use As Library

Assuming the Raggify server is already running, the following program can be executed, for example:

```python
import json

from raggify_client import RestAPIClient

client = RestAPIClient(host="localhost", port=8000)

print(json.dumps(client.status(), indent=2))

client.ingest_url("http://some.site.com")
```

# ⌨️ Use As CLI

Assuming the Raggify server is already running, you can execute commands such as the following:

<img alt="Image" src="https://raw.githubusercontent.com/jun76/raggify/main/media/client.png" />
