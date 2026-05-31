import logging
import os
from typing import Literal, Union
import shutil

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
import pandas as pd
import plotly.express as px

from agentic_valence.config import settings
from agentic_valence.prompts import load_prompt

logger = logging.getLogger()

PROMPT_VIZ_CREATOR = load_prompt("viz_creator")



def clean_artifacts():
    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")
    else:
        # Move old artifacts
        items = [i for i in os.listdir("artifacts")]
        old_artifact_dirs = [i for i in os.listdir(".") if i.startswith("artifacts_")]
        if items:
            a = max([0] + [int(i.split("_")[-1]) for i in old_artifact_dirs])
            shutil.move("artifacts", f"artifacts_{a+1}")



@tool
def create_interactive_plot(
    data: dict[str, list[Union[str, float]]],
    plot_type: Literal["line", "bar", "scatter", "histogram", "box", "area"] = "line",
    x: str = None,
    y: str | list[str] = None,
    color: str = None,
    title: str = "Interactive Plot",
) -> bool:
    """
    Creates an interactive Plotly figure from a pandas DataFrame.

    Args:
        data:      dictionary with keys being columns names and values being column values
        plot_type: One of 'line', 'bar', 'scatter', 'histogram', 'box', 'area'.
        x:         Column name for the x-axis (uses index if None).
        y:         Column name(s) for the y-axis (uses all numeric cols if None).
        color:     Column name used to color-code the series.
        title:     Chart title.

    Returns:
        True if success.
    """
    try:
        df = pd.DataFrame(data)

        # Fall back to numeric columns when y is not specified
        if y is None:
            y = df.select_dtypes(include="number").columns.tolist()

        # Use index as x-axis if not specified
        if x is None:
            df = df.copy()
            df["__index__"] = df.index
            x = "__index__"

        plot_fn = {
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "histogram": px.histogram,
            "box": px.box,
            "area": px.area,
        }

        if plot_type not in plot_fn:
            raise ValueError(
                f"Unsupported plot_type '{plot_type}'. " f"Choose from: {list(plot_fn)}"
            )

        kwargs = dict(data_frame=df, x=x, title=title)

        # histogram only accepts a single column for x — skip y/color
        if plot_type == "histogram":
            kwargs["x"] = y[0] if isinstance(y, list) else y
            if color:
                kwargs["color"] = color
        else:
            kwargs["y"] = y
            if color:
                kwargs["color"] = color

        fig = plot_fn[plot_type](**kwargs)

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        # Save to artifacts
        clean_artifacts()
        figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
        fig.write_html(f"artifacts/fig{len(figures)}.html")
        return True
    
    except Exception:
        return False

# @tool
# def create_molecule_plot(
#     basis: Any,
#     matrix_ao_mo: np.ndarray,
#     index: int,
#     isovalue: float = 0.07,
#     title: str = "Molecule"
# ):
#     raise NotImplementedError


model_viz_creator = ChatOpenAI(
    temperature=0,
    model_name=settings.model_viz_creator,
    api_key=settings.openai_key,
)

viz_creator = create_agent(
    model_viz_creator,
    tools=[create_interactive_plot],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_VIZ_CREATOR,
            }
        ]
    ),
)
