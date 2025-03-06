import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os

def parse_log_file(log_file_path):
    """Parse the log file and extract JSON data."""
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Extract JSON objects from the log file
    json_pattern = r'\{.*\}'
    json_matches = re.findall(json_pattern, content)
    
    data = []
    for json_str in json_matches:
        try:
            data.append(json.loads(json_str))
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_str[:50]}...")
    
    return data

def plot_metrics(data, output_dir='.'):
    """Plot various metrics from the parsed data using Plotly."""
    # Extract epochs - if not available, use indices
    epochs = list(range(len(data)))
    
    # Create figure with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Loss', 
            'Component Losses', 
            'Learning Rate', 
            'Component Losses (Detail)'
        )
    )
    
    # Plot 1: Training Loss
    train_loss = [entry.get('train_loss') for entry in data if 'train_loss' in entry]
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, mode='lines', name='Total Loss', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Plot 2: Component Losses
    loss_mal = [entry.get('train_loss_mal') for entry in data if 'train_loss_mal' in entry]
    loss_bbox = [entry.get('train_loss_bbox') for entry in data if 'train_loss_bbox' in entry]
    loss_giou = [entry.get('train_loss_giou') for entry in data if 'train_loss_giou' in entry]
    loss_fgl = [entry.get('train_loss_fgl') for entry in data if 'train_loss_fgl' in entry]
    
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_mal, mode='lines', name='MAL Loss', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_bbox, mode='lines', name='BBox Loss', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_giou, mode='lines', name='GIoU Loss', line=dict(color='purple')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_fgl, mode='lines', name='FGL Loss', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Plot 3: Learning Rate
    lr = [entry.get('train_lr') for entry in data if 'train_lr' in entry]
    fig.add_trace(
        go.Scatter(x=epochs, y=lr, mode='lines', name='Learning Rate', line=dict(color='cyan')),
        row=2, col=1
    )
    
    # Plot 4: Detailed Component Losses (Auxiliary)
    loss_mal_aux_0 = [entry.get('train_loss_mal_aux_0') for entry in data if 'train_loss_mal_aux_0' in entry]
    loss_bbox_aux_0 = [entry.get('train_loss_bbox_aux_0') for entry in data if 'train_loss_bbox_aux_0' in entry]
    
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_mal_aux_0, mode='lines', name='MAL Aux 0', line=dict(color='pink')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_bbox_aux_0, mode='lines', name='BBox Aux 0', line=dict(color='lightgreen')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Training Metrics',
        height=800,
        width=1200,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=2)
    
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=2)
    fig.update_yaxes(title_text='Learning Rate', row=2, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HTML file for interactivity
    html_path = os.path.join(output_dir, 'training_metrics.html')
    fig.write_html(html_path)
    
    # Also save as static image
    png_path = os.path.join(output_dir, 'training_metrics.png')
    fig.write_image(png_path, scale=2)
    
    print(f"Saved interactive plot to: {html_path}")
    print(f"Saved static image to: {png_path}")
    
    # Show the plot
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from log file')
    parser.add_argument('--log-file', type=str, default='outputs/deim_hgnetv2_n_coco/log.txt',
                        help='Path to the log file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output plots (defaults to log file directory)')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at {args.log_file}")
        return
    
    # If output_dir is not specified, use the directory of the log file
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.log_file))
        print(f"Output directory not specified, using log file directory: {args.output_dir}")
    
    data = parse_log_file(args.log_file)
    if not data:
        print("No valid data found in log file")
        return
    
    print(f"Processed {len(data)} data points from log file")
    plot_metrics(data, args.output_dir)

if __name__ == "__main__":
    main()
