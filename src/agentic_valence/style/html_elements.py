import os

import html
import json
from langchain.messages import AIMessage, ToolMessage


CSS_STYLE_HEADER = """<style>
        .milestone {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
            padding: 10px;
            background-color: #fafafa;
        }
        .milestone summary {
            font-weight: bold;
            cursor: pointer;
            font-size: 14px;
        }
        .milestone ul {
            margin-top: 8px;
        }
        .event-container { 
            word-wrap: break-word; 
            overflow-wrap: break-word; 
            white-space: normal;
            font-family: sans-serif;
            margin-bottom: 15px;
            border: 1px solid #bbf7d0; 
            padding: 12px; 
            border-radius: 8px; 
        }
        .code-block {
            white-space: pre-wrap; 
            word-break: break-all;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
    </style>
    """

def format_event_panel(new_event: AIMessage | ToolMessage, old_event_panel: str) -> str:

    # Get clean events
    new_event_html = parse_event_to_html(new_event)
    if new_event_html:
        return f"""{old_event_panel}<br>{new_event_html}"""
    return old_event_panel


def parse_event_to_html(event: AIMessage | ToolMessage) -> str:
    """
    Parses a LangChain event to HTML with dark mode support and overflow protection.
    """
    content = getattr(event, "content", "")
    msg_type = type(event).__name__

    # CSS for responsiveness and theme adaptation
    html_output = ""

    if msg_type == "AIMessage":
        tool_calls = getattr(event, "tool_calls", [])
        if isinstance(content, str):
            if not content.startswith("#"):
                html_output += f"""<div class="milestone">
                        <details>
                            <summary>🧠 Strategizing</summary>
                            <ul>{content}</ul>
                        </details>
                    </div>"""
        if tool_calls:
            for tc in tool_calls:
                name = html.escape(tc.get("name", "unknown"))
                args = html.escape(json.dumps(tc.get("args", {}), indent=2))
                html_output += f"""<div class="milestone">
                        <details>
                            <summary>🛠️ Action: <code>{name}</code></summary>
                            <ul><b>Task:</b><pre class="code-block"><code>{args}</code></pre></ul>
                        </details>
                    </div>"""


    if msg_type == "ToolMessage":
        # Escape the name for safety, but allow 'content' to render as raw HTML
        name = html.escape(getattr(event, "name", "Expert"))

        if name == "VizCreator":
            figures = sorted([
                i for i in os.listdir("artifacts")
                if i.startswith("fig") and (i.endswith("html") or i.endswith("png"))
            ])

            html_output = (
                f'<div class="event-container">'
                f'<h4 style="margin: 0 0 8px 0;">🎨 Figure {len(figures)}</h4>'
                f"""<iframe src="gradio_api/file/artifacts/{figures[-1]}" width="100%" height="500px"></iframe>"""
                f"</div>"
            )

        else:
            html_output += f"""<div class="milestone">
                        <details>
                            <summary>🧱 <code>{name}</code> results</summary>
                            <ul>{content}</ul>
                        </details>
                    </div>"""

    return f"{html_output.strip()}" if html_output else ""