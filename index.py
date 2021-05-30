import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import Home, multipleGraph,machineLearning


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='output-data-upload'),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/Home':
        return Home.layout
    elif pathname == '/apps/multipleGraph':
        return multipleGraph.layout
    elif pathname == '/apps/machineLearning':
        return machineLearning.layout
    else:
        return Home.layout

if __name__ == '__main__':
    app.run_server(debug=False)