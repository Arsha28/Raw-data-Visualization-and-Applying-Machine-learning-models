import dash  # version 1.13.1
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np
from app import app

df = pd.read_csv(r'C:\Users\ARSHAD\Desktop\Project_data\Data_Project2\upload\filenames.csv')



layout = html.Div([
    html.Div(children=[
        html.Button('Add Chart', id='add-chart', n_clicks=0),
    ]),
    html.Div(id='container', children=[])
])


@app.callback(
    Output('container', 'children'),
    [Input('add-chart', 'n_clicks')],
    [State('container', 'children')]
)
def display_graphs(n_clicks, div_children):
    new_child = html.Div(
        style={'width': '45%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
        children=[
           dcc.Loading(dcc.Graph(
                id={
                    'type': 'dynamic-graph',
                    'index': n_clicks
                },
                figure={}
            ),className='graphLoading'),
            html.Div([
            dcc.RadioItems(
                id={
                    'type': 'chart_choice',
                    'index': n_clicks
                },
                
                options=[{'label': 'Bar Chart', 'value': 'bar'},
                         {'label': 'Line Chart', 'value': 'line'},
                         {'label': 'Pie Chart', 'value': 'pie'},
                         {'label': 'Scatter Chart', 'value': 'scatter'},
                         {'label': 'Histogram', 'value': 'histogram'}],
                value = 'bar',
                labelStyle={'display': 'block'}
            ),],className='radio3'),
            dcc.Dropdown(
                id={
                    'type': 'xaxis-column',
                    'index': n_clicks
                },
               
                options=[{'label': i, 'value': i} for i in df],
                placeholder='X-axis',
                
            ),
            dcc.Dropdown(
                id={
                    'type': 'yaxis-column',
                    'index': n_clicks
                },
                
                options=[{'label': i, 'value': i} for i in df],
                placeholder='Y-axis',
            ),
        ]
    )
    div_children.append(new_child)
    return div_children


@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'xaxis-column', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'yaxis-column', 'index': MATCH}, component_property='value'),
     Input({'type': 'chart_choice', 'index': MATCH}, component_property='value')]
)
def update_graph(xaxis_column_name,yaxis_column_name,chart_choice):
  
    if chart_choice =='bar':
        fig = px.bar(df,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
        fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgb(37, 41, 84)',
                    font_color='#fff',
                    plot_bgcolor='#252954',
                    
            )
        return fig
    elif chart_choice =='line':
        fig = px.line(df,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
        fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgb(37, 41, 84)',
                    font_color='#fff',
                    plot_bgcolor='#252954'
                    
            )
        return fig
    elif chart_choice == 'pie':
        fig = px.pie(df,names=xaxis_column_name,values=yaxis_column_name,
                        hole = 0.7)
        fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgb(37, 41, 84)',
                    font_color='#fff',
                    plot_bgcolor='#252954',
                    
            )
        return fig
    elif chart_choice =='scatter':
        fig = px.scatter(df,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
        fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgb(37, 41, 84)',
                    font_color='#fff',
                    plot_bgcolor='#252954',
                
            )
        return fig
    elif chart_choice =='histogram':
        fig = px.histogram(df,x=xaxis_column_name,y=yaxis_column_name,nbins=10)
        fig.update_layout(
                    
                    paper_bgcolor='rgb(37, 41, 84)',
                    font_color='#fff',
                    plot_bgcolor='#252954',
                    
            )
        return fig


