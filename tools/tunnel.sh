#!/usr/bin/env bash
# Routes microopus.fueldeskpro.com through the fueldeskpro-mac tunnel to this app. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="microopus.fueldeskpro.com"
TUNNEL="fueldeskpro-mac"
PORT="$(cat PORT)"
CFG="/etc/cloudflared/config.yml"

if grep -q "hostname: $HOST" "$CFG"; then
  echo "Ingress rule for $HOST already present in $CFG"
else
  sudo cp "$CFG" "$CFG.bak.$(date +%Y%m%d%H%M%S)"
  sudo /usr/bin/python3 - "$CFG" "$HOST" "$PORT" <<'PY'
import re, sys
path, host, port = sys.argv[1:]
text = open(path).read()
m = re.search(r"^([ \t]*)- service:\s*http_status:404", text, re.M)
if not m:
    sys.exit("No catch-all 'service: http_status:404' rule found; add the rule by hand.")
ind = m.group(1)
rule = f"{ind}- hostname: {host}\n{ind}  service: http://localhost:{port}\n"
open(path, "w").write(text[:m.start()] + rule + text[m.start():])
print(f"Added {host} -> localhost:{port}")
PY
fi

cloudflared tunnel --config "$CFG" ingress validate
cloudflared tunnel route dns "$TUNNEL" "$HOST" || echo "DNS route not created here. Add a CNAME for microopus pointing to the $TUNNEL tunnel in the Cloudflare dashboard."
sudo launchctl kickstart -k system/com.cloudflare.cloudflared
sleep 4
curl -fsS "https://$HOST/api/health" && echo && echo "Live at https://$HOST" || echo "Not reachable yet. DNS can take a minute; retry: curl https://$HOST/api/health"
