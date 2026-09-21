@echo off
rem Launch the MCP Photoshop server over streamable-HTTP for Open WebUI.
rem
rem - Binds 127.0.0.1:8000 (FastMCP default). Docker Desktop NATs
rem   host.docker.internal:8000 -> host loopback, so the Open WebUI container
rem   connects with URL: http://host.docker.internal:8000/mcp
rem - Cline's stdio connection is unaffected (it launches its own process
rem   with MCP_TRANSPORT unset -> stdio).
rem - Open WebUI (admin) -> Integrations -> External Tool Servers -> + :
rem   Type "MCP (Streamable HTTP)", URL http://host.docker.internal:8000/mcp,
rem   Auth: None.
rem - Logs append to mcp_http.log in this folder (the script's own folder;
rem   this file stores no absolute paths).
set MCP_TRANSPORT=streamable-http
cd /d "%~dp0"
python server.py >> mcp_http.log 2>&1
pause
