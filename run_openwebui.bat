@echo off
rem Launch the MCP Photoshop server over streamable-HTTP for Open WebUI.
rem
rem - Binds 0.0.0.0:8000 (MCP_HTTP_HOST below). Docker Desktop NATs
rem   host.docker.internal:8000 -> host loopback, so the Open WebUI container
rem   connects with URL: http://host.docker.internal:8000/mcp, and Tailscale
rem   tailnet peers connect with their tailnet IP / MagicDNS host on port 8000.
rem   server.py's transport allowlist (tailnet patterns come from
rem   MCP_HTTP_ALLOWED_HOSTS in .env) rejects all other Host/Origin headers.
rem - Cline's stdio connection is unaffected (it launches its own process
rem   with MCP_TRANSPORT unset -> stdio).
rem - Open WebUI (admin) -> Integrations -> External Tool Servers -> + :
rem   Type "MCP (Streamable HTTP)", URL http://host.docker.internal:8000/mcp,
rem   Auth: None.
rem - Logs append to mcp_http.log in this folder (the script's own folder;
rem   this file stores no absolute paths).
set MCP_TRANSPORT=streamable-http
rem Bind all interfaces so tailnet peers can reach :8000 (see comment above).
set MCP_HTTP_HOST=0.0.0.0
cd /d "%~dp0"
python server.py >> mcp_http.log 2>&1
pause
