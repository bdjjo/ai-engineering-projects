"""Gradio viewer for precomputed measurements. Does not load model weights."""

import html
import json
from pathlib import Path


def render_feature(index, feature_id, limit=4):
    numeric = float(feature_id)
    if not numeric.is_integer() or not 0 <= numeric < index["hidden_dim"]:
        raise ValueError(f"Feature must be an integer from 0 to {index['hidden_dim'] - 1}")
    feature = index["features"][int(numeric)]
    title = f"### Feature {feature['id']}\n"
    title += f"Active on **{feature['active_fraction']:.1%}** of probe tokens. "
    title += f"Maximum activation: **{feature['max_activation']:.3f}**.\n\n"
    title += (
        "Examples are ranked within this small probe corpus; a semantic concept is not established."
    )
    blocks = []
    for example in feature["examples"][: int(limit)]:
        context = (
            html.escape(example["before"])
            + "<mark>"
            + html.escape(example["token"])
            + "</mark>"
            + html.escape(example["after"])
        )
        blocks.append(f"<p><strong>{example['activation']:.3f}</strong> &nbsp; {context}</p>")
    if not blocks:
        blocks.append(
            "<p>No positive activation in this probe corpus. "
            "This does not prove a dead feature.</p>"
        )
    return title, "".join(blocks)


def build_browser(index_path):
    import gradio as gr

    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    default = max(index["features"], key=lambda f: f["max_activation"])["id"]
    with gr.Blocks(title="GPT-2 Feature Explorer") as demo:
        gr.Markdown(
            "# GPT-2 Feature Explorer\n"
            "Inspect sparse-autoencoder activations in short text contexts. "
            "The highlighted token is where the feature activated."
        )
        gr.Markdown(
            f"**{index['hidden_dim']:,} features** · **{index['probe_tokens']:,} probe tokens** "
            f"· `{index['hook_name']}`\n\n"
            "A small training experiment and an exploratory probe set. "
            "Feature meanings have not been independently validated."
        )
        feature_id = gr.Number(
            label=f"Feature index (0–{index['hidden_dim'] - 1})", value=default, precision=0
        )
        count = gr.Slider(1, 4, value=4, step=1, label="Examples")
        button = gr.Button("Inspect feature", variant="primary")
        initial_title, initial_examples = render_feature(index, default)
        summary = gr.Markdown(initial_title)
        examples = gr.HTML(initial_examples)

        def inspect(feature, n):
            try:
                return render_feature(index, feature, n)
            except (ValueError, TypeError, OverflowError) as error:
                raise gr.Error(str(error)) from error

        button.click(inspect, [feature_id, count], [summary, examples], api_name="inspect")
        gr.Markdown(
            "[Source and measured results](https://github.com/bdjjo/ai-engineering-projects/"
            "tree/main/projects/03-mechanistic-interpretability). "
            "Probe paragraphs were written for this project; no user input is sent to a model."
        )
    return demo
