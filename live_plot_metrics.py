import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Live plot training metrics from a log file.')
parser.add_argument('--log-file', type=str, default='outputs/deim_hgnetv2_n/log.txt',
                    help='Path to the log file containing training metrics')
parser.add_argument('--max-points', type=int, default=300,
                    help='Maximum number of data points to show in the plot')
parser.add_argument('--update-interval', type=int, default=1000,
                    help='Update interval in milliseconds')

args = parser.parse_args()

# Use the arguments
DATA_FILE = args.log_file
MAX_POINTS = args.max_points
UPDATE_INTERVAL = args.update_interval

def parse_log_file(log_file_path):
    """Parse the log file and extract JSON data."""
    if not os.path.exists(log_file_path):
        print(f"Warning: File {log_file_path} does not exist")
        return []
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Extract JSON objects from the log file
        json_pattern = r'\{.*\}'
        json_matches = re.findall(json_pattern, content)
        
        data = []
        for i, json_str in enumerate(json_matches):
            try:
                parsed = json.loads(json_str)
                # Add an index for the x-axis
                parsed['step'] = i
                data.append(parsed)
            except json.JSONDecodeError:
                continue
        
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Training Metrics Live Plot"),
    
    html.Div([
        html.Label("Select metrics to plot:"),
        dcc.Dropdown(
            id='metrics-dropdown',
            options=[
                {'label': 'Learning Rate', 'value': 'train_lr'},
                {'label': 'Total Loss', 'value': 'train_loss'},
                {'label': 'MAL Loss', 'value': 'train_loss_mal'},
                {'label': 'BBox Loss', 'value': 'train_loss_bbox'},
                {'label': 'GIoU Loss', 'value': 'train_loss_giou'},
                {'label': 'FGL Loss', 'value': 'train_loss_fgl'}
            ],
            value=['train_loss', 'train_loss_mal', 'train_loss_bbox'],
            multi=True
        )
    ], style={'width': '50%', 'margin': '10px'}),
    
    # Training metrics plot
    html.Div([
        html.H3("Training Metrics"),
        dcc.Graph(id='training-graph')
    ]),
    
    # Learning Rate plot
    html.Div([
        html.H3("Learning Rate"),
        dcc.Graph(id='lr-graph')
    ]),
    
    # AP and AR plots in a row
    html.Div([
        html.Div([
            html.H3("Average Precision (AP @IoU=0.50:0.95)"),
            dcc.Graph(id='ap-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Average Recall (AR @IoU=0.50:0.95)"),
            dcc.Graph(id='ar-graph')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL,
        n_intervals=0
    )
])

@app.callback(
    [Output('training-graph', 'figure'),
     Output('lr-graph', 'figure'),
     Output('ap-graph', 'figure'),
     Output('ar-graph', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('metrics-dropdown', 'value')]
)
def update_graphs(n, selected_metrics):
    """Update all graphs with new data."""
    # Read the data
    data = parse_log_file(DATA_FILE)
    
    if not data:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Extract epochs/steps
    epochs = [entry.get('step') for entry in data]
    
    # Create figures for each tab
    training_fig = make_subplots(rows=1, cols=1)
    lr_fig = make_subplots(rows=1, cols=1)
    ap_fig = make_subplots(rows=1, cols=1)
    ar_fig = make_subplots(rows=1, cols=1)
    
    # Add traces for training metrics (excluding learning rate)
    for metric in selected_metrics:
        if metric != 'train_lr':  # Skip learning rate for the training metrics plot
            metric_values = [entry.get(metric) for entry in data if metric in entry]
            if metric_values:
                training_fig.add_trace(
                    go.Scatter(
                        x=epochs[:len(metric_values)], 
                        y=metric_values,
                        mode='lines+markers',
                        name=metric
                    )
                )
    
    # Add learning rate trace to its own plot
    lr_values = [entry.get('train_lr') for entry in data if 'train_lr' in entry]
    if lr_values:
        lr_fig.add_trace(
            go.Scatter(
                x=epochs[:len(lr_values)], 
                y=lr_values,
                mode='lines+markers',
                name='Learning Rate',
                line=dict(color='green'),
                marker=dict(size=6)
            )
        )
    
    # Extract AP and AR values
    map_values = []
    map_epochs = []
    ar_values = []
    ar_epochs = []
    
    for i, entry in enumerate(data):
        if 'test_coco_eval_bbox' in entry and entry['test_coco_eval_bbox']:
            try:
                # Extract the first value (mAP 50:95) from the test_coco_eval_bbox list
                map_value = entry['test_coco_eval_bbox'][0]
                map_values.append(map_value)
                map_epochs.append(epochs[i])
                
                # Extract the 7th value (AR @IoU=0.50:0.95) from the test_coco_eval_bbox list
                if len(entry['test_coco_eval_bbox']) >= 7:
                    ar_value = entry['test_coco_eval_bbox'][6]  # 7th value (index 6)
                    ar_values.append(ar_value)
                    ar_epochs.append(epochs[i])
            except (IndexError, TypeError):
                continue
    
    # Add AP trace
    if map_values:
        ap_fig.add_trace(
            go.Scatter(
                x=map_epochs, 
                y=map_values,
                mode='lines+markers',
                name='AP @IoU=0.50:0.95',
                line=dict(color='darkblue'),
                marker=dict(size=8)
            )
        )
    
    # Add AR trace
    if ar_values:
        ar_fig.add_trace(
            go.Scatter(
                x=ar_epochs, 
                y=ar_values,
                mode='lines+markers',
                name='AR @IoU=0.50:0.95',
                line=dict(color='darkred'),
                marker=dict(size=8)
            )
        )
    
    # Update layouts
    training_fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Value",
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision='constant',
        height=400
    )
    
    lr_fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        margin=dict(l=50, r=50, t=30, b=50),
        uirevision='constant',
        height=300
    )
    
    ap_fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="AP",
        margin=dict(l=50, r=50, t=30, b=50),
        uirevision='constant',
        height=300
    )
    
    ar_fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="AR",
        margin=dict(l=50, r=50, t=30, b=50),
        uirevision='constant',
        height=300
    )
    
    return training_fig, lr_fig, ap_fig, ar_fig

if __name__ == '__main__':
    print(f"Starting live plot server. Reading data from {DATA_FILE}")
    print("Open http://127.0.0.1:8050/ in your browser to view the plot")
    app.run_server(debug=True)
