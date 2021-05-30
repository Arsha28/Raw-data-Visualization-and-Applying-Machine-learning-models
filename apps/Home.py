from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import io
import os
import pandas as pd
import dash
import dash_table
import plotly.express as px
import plotly.graph_objects as go


from app import app
app.config.suppress_callback_exceptions = True

layout = html.Div([
    
    # This is the 'hidden div' however its really a container for sub-divs, some hidden, some not 
    
    
    html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
           # 'Drag and Drop',
            html.A('Upload File',className='up')
        ],
        
    )),
    dcc.Markdown('''
       **Note**: 
       * After selecting the file click __*Submit*__ button.
       * If you want to upload another file, you dont need to click _**Submit**_ again
       it will submit automatically.
       * Please upload .csv extention file.

     ''',id='Markdown'),
     html.Button('Submit', id='btn'),
    ],className='uploadBox'),
    
    
    
    
    html.Div(id='output-container-button'),
    html.Div(id='class')
   
    
    
    
])





def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    df.to_csv(r'C:\Users\ARSHAD\Desktop\Project_data\Data_project2\upload\filenames.csv')
    return filename,df
    


#################################################################################################



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),])
def update_output(contents, filename):
    if contents is not None:
        filename, df = parse_contents(contents, filename)
        # This is the key to the hidden div formatting
        return html.Div([
                html.Div(['The ' + filename+' file has been selected'],id='file'),
                html.Div([df.to_json(orient = 'split')], id='tankStats',style={'display': 'none'})
                ],style={'padding-top': '60px','align': "center"})



#################################################################################################

@app.callback(Output('output-container-button', 'children'),
              [Input('btn', 'n_clicks'),
               Input('tankStats', 'children')])

def update_graph(n_clicks,tankStats):
    if n_clicks is not None:
        dff = pd.read_json(tankStats[0], orient='split')
        #figure = create_figure(dff)
        index = dff.index
        rows=len(index)
        rs = '{rows}'.format(rows=rows)
        col=dff.columns
        colu=len(col)
        cells=colu*rows
        cs = '{cells}'.format(cells=cells)
        return html.Div([
        html.Div(['There are total '+ rs+'rows ('+cs+' cells)  have been successfully parsed, now you can choose a chart'],id='rows_data'),
        
        html.Div([
            dcc.Link('Multi Graph', href='/apps/multipleGraph',id='multiGraph'),
            dcc.Link('Machine', href='/apps/machineLearning',id='machineLearning'),
            
            
            dcc.Markdown(''' 
        To view multiple graph visulization 
        click below  __*MultiGraph*__ button.
       

     ''',id='multiMark'),
     dcc.Markdown(''' 
        To apply the machine Learning models
        click below  __*Machine*__ button.
       

     ''',id='machineMark'),

        
        
        ],className='links'),
        
        dcc.Loading( dash_table.DataTable(
            id='Tabel',
            data=dff.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in dff.columns],
            style_cell=dict(textAlign='left'),
            style_header={
        'background-color': 'rgba(255,255,255,0.3)',
        'fontWeight': 'bold',
        'border-collapse': 'collapse',
        
        'text-transform': 'uppercase'
    },
            style_data={
        'background-color': 'rgba(255,255,255,0.3)',
        'fontWeight': 'bold',
        'border-collapse': 'collapse',
        

            }
            
        )),
        
        html.Div([
        html.Div([
        
            
        html.Div([
            
            dcc.Dropdown(
               
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in dff],
                placeholder='X-axis',
                
                
            ),
        
        
            
        ],className='xAxis',
        ),

        html.Div([
            dcc.Dropdown(
                
                
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in dff],
                placeholder='Y-axis',
                
              
                
            ),
           
            
        ], className='yAxis'),
     
    ]),
    html.Div([
        html.Div([
           
    dcc.RadioItems(
                id='chart-choice',
                options=[{'label': 'Bar Chart', 'value': 'bar'},
                         {'label': 'Line Chart', 'value': 'line'},
                         {'label': 'Pie Chart', 'value': 'pie'},
                         {'label': 'Scatter Chart', 'value': 'scatter'},
                         {'label': 'Histogram', 'value': 'histogram'}],
                value = 'bar',
                labelStyle={'display': 'block'}

               
            ),
        ],className='radio'),
    
       
   
    dcc.Loading(dcc.Graph(id='indicator-graphic',config={
                            'displayModeBar': True,
                        } 
    ),className='graphLoading'),
    
   ]),
        ],className='graphBox'),



       
    ])






#################################################################################################

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('chart-choice','value'),
    Input('btn', 'n_clicks'),
    Input('tankStats', 'children')
    ])


def update_graph(xaxis_column_name,yaxis_column_name,chart_choice,n_clicks,tankStats):
    if n_clicks is not None:
        dff = pd.read_json(tankStats[0], orient='split')
  
        if chart_choice=='bar':
            fig = px.bar(dff,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        
                )
            return fig
        elif chart_choice=='line':
            fig = px.line(dff,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954'
                        
                )
            return fig
        elif chart_choice == 'pie':
            fig = px.pie(dff,names=xaxis_column_name,values=yaxis_column_name,hole = .5)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        
                )
            return fig
        elif chart_choice=='scatter':
            fig = px.scatter(dff,x=xaxis_column_name,y=yaxis_column_name,color=xaxis_column_name)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                    
                )
            return fig
        elif chart_choice=='histogram':
            fig = px.histogram(dff,x=xaxis_column_name,y=yaxis_column_name,nbins=10)
            fig.update_layout(
                        
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        
                )
            return fig






