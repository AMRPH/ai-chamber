"""Run as root on tantir, after reviewing the prepared unit and route."""
from pathlib import Path
import shutil
import subprocess
import time
root=Path(__file__).resolve().parent
config=Path('/etc/nginx/tantir.conf')
source=config.read_text()
route=(root/'nginx-location.conf').read_text()
if 'location /ai-chamber/' not in source:
    backup=config.with_name('tantir.conf.before-ai-chamber-'+time.strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(config,backup)
    # Only the existing tantir.net HTTPS server receives the new location.
    anchor='    server_name tantir.net;'
    assert anchor in source
    assert source.index(anchor) < source.index('listen 443 ssl;')
    config.write_text(source.replace(anchor,anchor+'\n\n'+route,1))
    try:
        subprocess.run(['nginx','-t'],check=True)
    except Exception:
        shutil.copy2(backup,config)
        raise
shutil.copy2(root/'ai-chamber.service','/etc/systemd/system/ai-chamber.service')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','enable','--now','ai-chamber.service'],check=True)
subprocess.run(['systemctl','reload','nginx'],check=True)
