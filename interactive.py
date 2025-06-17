"""
python interactive.py
python interactive.py --text "The quick brown fox jumps over the lazy dog."
python interactive.py --file ./my_text_file.txt
python interactive.py --model "gpt2" --top-k 5 --text "Hello world"
python interactive.py --text "Hello world" --output output.html
"""

import argparse
import sys
import torch
import transformers
import plotly.graph_objects as go
import numpy as np

def visualize_token_predictions(model_name: str, input_text: str, top_k: int, output_file: str = None):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    inp_for_logits = inputs.input_ids[:, :-1]
    target_outp = inputs.input_ids[:, 1:].unsqueeze(-1)

    if inp_for_logits.shape[1] == 0:
        print("Input text is too short to process. It must result in at least 2 tokens.", file=sys.stderr)
        return

    with torch.no_grad():
        logits = model(inp_for_logits).logits.float()

    all_probs = torch.softmax(logits, dim=-1)
    chosen_probs = torch.gather(all_probs, dim=2, index=target_outp).squeeze(-1).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())

    tokens = tokens[1:]
    
    top_k_probs, top_k_indices = torch.topk(all_probs, top_k, dim=-1)

    hover_texts = []
    for i in range(len(tokens)):
        indices = top_k_indices[0, i, :].cpu().tolist()
        probabilities = top_k_probs[0, i, :].cpu().tolist()
        decoded_tokens = tokenizer.convert_ids_to_tokens(indices)
        
        display_token = tokens[i].replace('Ġ', ' ')
        
        hover_str = f"<b>Token:</b> {repr(display_token)}<br>"
        hover_str += f"<b>Probability:</b> {chosen_probs[i]:.2%}<br><br>"
        hover_str += f"<b>Top {top_k} Predictions:</b><br>"
        
        for j, (token, prob) in enumerate(zip(decoded_tokens, probabilities)):
            clean_token = token.replace('Ġ', ' ')
            if indices[j] == target_outp[0, i, 0].item():
                hover_str += f"<b>→ {repr(clean_token)}: {prob:.2%}</b><br>"
            else:
                hover_str += f"  {repr(clean_token)}: {prob:.2%}<br>"
        
        hover_texts.append(hover_str)

    tokens_per_row = 20
    num_rows = (len(tokens) + tokens_per_row - 1) // tokens_per_row
    
    grid_tokens, grid_probs, grid_hover = [], [], []
    for i in range(num_rows):
        row_start = i * tokens_per_row
        row_end = row_start + tokens_per_row
        
        row_tokens_slice = tokens[row_start:row_end]
        row_probs_slice = chosen_probs[row_start:row_end]
        row_hover_slice = hover_texts[row_start:row_end]

        display_tokens = [t.replace('Ġ', ' ').replace(' ', '␣') for t in row_tokens_slice]
        
        padding_needed = tokens_per_row - len(row_tokens_slice)
        grid_tokens.append(display_tokens + [''] * padding_needed)
        grid_probs.append(list(row_probs_slice) + [np.nan] * padding_needed)
        grid_hover.append(row_hover_slice + [''] * padding_needed)
    
    fig = go.Figure(data=go.Heatmap(
        z=grid_probs,
        text=grid_tokens,
        customdata=grid_hover,
        texttemplate='%{text}',
        textfont={"size": 12, "family": "monospace"},
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title=dict(
                text="Token Probability",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            tickformat=".0%"
        ),
        hovertemplate='%{customdata}<extra></extra>',
        showscale=True,
        xgap=2,
        ygap=2
    ))
    
    fig.update_layout(
        title=f"Token Predictions Visualization - Model: {model_name}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        plot_bgcolor='white',
        height=100 + 40 * num_rows,
        width=1200,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    if output_file:
        print(f"Saving visualization to {output_file}...")
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"Interactive visualization saved as '{output_file}'")
    else:
        print("Opening visualization in browser...")
        fig.show()

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Token Probability and Prediction Visualizer using Plotly.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-t", "--text",
        type=str,
    )
    group.add_argument(
        "-f", "--file",
        type=str,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.html",
    )
    
    args = parser.parse_args()

    if args.text:
        input_text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        print("No input text or file provided. Using the default example text.")
        input_text = "The quick brown fox jumps over the lazy dog."

    visualize_token_predictions(
        model_name=args.model,
        input_text=input_text,
        top_k=args.top_k,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
