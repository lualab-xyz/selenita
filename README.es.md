# selenita

[English](README.md) · **Español**

Agente de IA que implementa el **[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) v1**,
construido sobre **[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (Python)**
y el SDK ACP oficial Python (`agent-client-protocol`).

Se integra con cualquier cliente ACP (p. ej. [Zed](https://zed.dev)) y también se usa
directamente por terminal.

## Instalación (canal alpha)

Las builds alpha se publican como pre-releases etiquetadas (`v0.1.0aN`, PEP 440) en el
repo público de distribución [lualab-xyz/selenita](https://github.com/lualab-xyz/selenita):

```bash
# Linux/macOS — una línea (descarga la wheel del release, verifica el SHA256SUMS,
# instala en ~/.selenita/venv y enlaza `selenita` en ~/.local/bin)
curl -fsSL https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.ps1 | iex

# O manualmente, en cualquier plataforma con Python >= 3.11
pip install https://github.com/lualab-xyz/selenita/releases/download/v0.1.0a1/selenita-0.1.0a1-py3-none-any.whl
```

Los releases los dispara el tag: al pushear `v0.1.0aN` al repo público se compila la
wheel + sdist, se hace un smoke del CLI instalado en las plataformas soportadas
(linux-x64, windows-x64) y se adjuntan wheel, sdist, instaladores y SHA256SUMS.

## Uso (núcleo LLM activo, con streaming del razonamiento y fallback a eco)

Los tres modos apuntan al mismo núcleo de agente (un solo core, tres adaptadores):

```bash
# 1) Servidor ACP sobre stdio (config de un cliente ACP: command=selenita, args=["--acp"])
selenita --acp

# 2) CLI one-shot (crea sesión nueva e imprime su session-id)
selenita "hola"           # → respuesta + [session: sess_xxx]
selenita --prompt "hola"  # equivalente (-p)

# 2b) CLI one-shot reanudando sesión persistida (JSONL en ~/.selenita/sessions)
selenita "más cosas" --session-id sess_8e1f...

# 3) TUI interactiva con la misma sesión (prompt ❯ cian, respuestas del agente con ¤,
#    razonamiento del modelo en streaming tenue con ·, Alt+Intro = línea nueva;
#    "exit"/quit o Ctrl+D/Ctrl+C salen; bienvenida con el último mensaje de la
#    sesión, recortado si es largo)
selenita
selenita --session-id sess_8e1f...
```

Nota: Alt+Intro y Shift+Intro insertan línea nueva si el terminal las distingue (secuencias
``ESC CR``, kitty ``CSI 13;<mod>u`` o xterm ``CSI 27;<mod>;13~``); en muchos emuladores
Shift+Intro equivale a Intro y solo Alt+Intro funciona.

## Configuración

Por variables de entorno (`SELENITA_*`) y, opcionalmente, por un fichero `.env` en el
directorio actual — las variables reales siempre ganan:

```bash
# .env (no se versiona; ver .gitignore)
SELENITA_AGENT_MODE=llm            # o "echo" para forzar el núcleo eco
SELENITA_LLM_PROVIDER=openai       # "openai" (Responses API) | "openai-chat" | "ollama"
SELENITA_OPENAI_BASE_URL=https://your-llm-endpoint.example/v1
SELENITA_OPENAI_MODEL=default
SELENITA_OPENAI_API_KEY=sk-your-key
```

> El cliente Responses corre siempre en modo stateless (`store=False`, historial inline —
> diseño D014); la base URL por defecto apunta a la única ruta del gateway que soporta bucles
> de tools en Responses. `previous_response_id` queda como optimización futura.

Si falta la configuración del LLM, la degradación es amable: selenita arranca en **modo
eco** y registra un warning; los turnos siguen funcionando.

```bash
# Servidor ACP sobre TCP, NDJSON (transporte custom, aún sin implementar)
selenita --acp --port 8765

# MCP en modo terminal: conectar UN servidor MCP stdio (atajo de servidor único)
selenita "Usa la tool add para calcular 2 + 3" --mcp-cmd "npx -y @modelcontextprotocol/server-filesystem ."

# ... o varios servidores por env (el name es opcional):
# SELENITA_MCP_SERVERS='[{"name":"demo","command":"python","args":["mcp_server.py"]}]'
```

## Stack

| Pieza | Elección |
|---|---|
| Lenguaje | Python 3.11+ (asyncio) |
| Núcleo de agente | Microsoft Agent Framework (`agent-framework-core`) |
| Protocolo | ACP v1 vía SDK Python oficial (`agent-client-protocol`) |
| MCP | MCPStdioTool de MAF (servers pasados por el cliente ACP y/o configurados localmente) |
| LLM | OpenAI (Responses API) y Ollama, configurable por env (`SELENITA_LLM_PROVIDER`) |
| Tests | pytest + pytest-asyncio, con LLM y MCP mockeados desde el día 1 |
| Calidad | ruff + mypy (src tipado) |

## Estado

🚧 **En construcción** — núcleo LLM activo (respuestas y razonamiento en streaming), fallback
eco, servidor `--acp`, CLI one-shot y TUI interactiva al mismo núcleo (74 tests). Documentación
viva en [`/.context/`](.context/) (interna, en español en su mayoría).

Flujo de trabajo: ramas `feature/task-name`, checkpoints frecuentes, nunca directamente sobre `main`/`dev`.