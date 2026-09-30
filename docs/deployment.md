# Deployment

Use the CUDA installation in the root README first. The examples below assume Linux with systemd, nginx and HTTPS already configured for your domain. No installer modifies your nginx configuration automatically.

## systemd

`deploy/ai-chamber.service` is a template for a checkout at `/opt/ai-chamber`, owned by the `ai-chamber` service account. If you use another account or directory, edit **User**, **Group**, **WorkingDirectory**, **EnvironmentFile** and **ExecStart** together. The service account must be able to read its weights and write its cache and `cuda-engine.log`.

For a new dedicated account:

```sh
sudo useradd --system --user-group --no-create-home --home-dir /opt/ai-chamber --shell /usr/sbin/nologin ai-chamber
sudo install -d -o ai-chamber -g ai-chamber /opt/ai-chamber
sudo -u ai-chamber git clone https://github.com/AMRPH/ai-chamber.git /opt/ai-chamber
sudo -u ai-chamber python3.13 -m venv /opt/ai-chamber/.venv
sudo -u ai-chamber /opt/ai-chamber/.venv/bin/python -m pip install -r /opt/ai-chamber/requirements-cuda.txt
sudo -u ai-chamber cp /opt/ai-chamber/.env.example /opt/ai-chamber/.env
sudo chmod 600 /opt/ai-chamber/.env
```

Edit `/opt/ai-chamber/.env` for your hardware, model and password digest. Configure any additional GPU device permissions required by your distribution. Then install the reviewed unit:

```sh
sudo cp deploy/ai-chamber.service /etc/systemd/system/ai-chamber.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-chamber.service
sudo journalctl -u ai-chamber.service -f
```

The default backend address is `127.0.0.1:18765`. Check readiness:

```sh
curl --fail http://127.0.0.1:18765/api/status
```

Expect `state: ready` and the configured model. During the initial weight download or load it will report `loading`. CUDA diagnostics are also written to `cuda-engine.log` in the checkout.

## nginx and HTTPS

Place the contents of `deploy/nginx-location.conf` **inside your existing HTTPS server block**. It exposes `/ai-chamber/` and redirects `/ai-chamber` to the trailing-slash URL. It forwards WebSockets and preserves the host used for same-origin checks. Change the port in both the unit and proxy configuration if necessary.

```sh
sudo nginx -t
sudo systemctl reload nginx
```

Open `https://YOUR_DOMAIN/ai-chamber/`. Obtain and maintain a TLS certificate using your existing server workflow before exposing the chat; model-switch passwords travel in HTTP headers. If the service should be private, restrict access in your reverse proxy. Model selection is global and password protected, while the chat endpoint has no user authentication.

The single worker owns one accelerator. Do not add Uvicorn workers: each would load a separate model and own a separate queue. CUDA defaults to up to eight admitted replies (`CHAMBER_PARALLEL=8`) and `CHAMBER_GPU_MEMORY=0.95`. Actual simultaneous long-context generation is limited by available KV-cache memory; vLLM schedules requests within that cache. Adjust these settings in `.env`, then restart the service. MPS/CPU defaults to two replies.

## Updates

Review the changes, stop the service, update the checkout and install its backend requirements before restarting. Keep your local `.env`, model cache and any research outputs; none are tracked in Git.

```sh
sudo systemctl stop ai-chamber.service
sudo -u ai-chamber git -C /opt/ai-chamber pull --ff-only
sudo -u ai-chamber /opt/ai-chamber/.venv/bin/python -m pip install -r /opt/ai-chamber/requirements-cuda.txt
sudo systemctl start ai-chamber.service
curl --fail http://127.0.0.1:18765/api/status
```

For problems, inspect the unit journal and `cuda-engine.log`; avoid posting your `.env` or credentials with an issue.
