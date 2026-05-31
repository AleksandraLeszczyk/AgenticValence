import os
import uuid

import gradio as gr

from agentic_valence.main import chat_with_principal_investigator
from agentic_valence.style.html_elements import CSS_STYLE_HEADER

# Importing agentic_valence.config (transitively via main) loads the .env file
# and validates settings, so no explicit load_dotenv() is needed here.


def _new_session_artifact_dir() -> str:
    path = os.path.join("artifacts", str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="AgenticValence", theme=theme) as ui:
        gr.Markdown("## 🏢 Quantum Chemistry Lab")

        # One artifact directory per browser session, created on first load.
        session_artifact_dir = gr.State(_new_session_artifact_dir)

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Lab",
                    height=600,
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": False}],
                )
                message = gr.Textbox(
                    label="Research Project",
                    placeholder="Ask anything about quantum chemistry...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                event_html = gr.HTML(
                    label="Computational Details",
                    value="Research steps and figures will appear here.",
                    css_template=CSS_STYLE_HEADER,
                    container=True,
                    height=650,
                    max_height=650,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat_with_principal_investigator,
            inputs=[chatbot, event_html, session_artifact_dir],
            outputs=[chatbot, event_html],
        )

    ui.launch(
        inbrowser=True,
        allowed_paths=["artifacts"],
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    main()
