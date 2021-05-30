import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report ,accuracy_score   
import plotly.figure_factory as ff  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.naive_bayes import GaussianNB 
import numpy as np
from sklearn import metrics
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from app import app
app.config.suppress_callback_exceptions = True

df = pd.read_csv(r'C:\Users\ARSHAD\Desktop\Project_data\Data_Project2\upload\filenames.csv')


layout = html.Div(children=[

    dcc.Loading( dash_table.DataTable(
            id='Tabel1',
            
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
        
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
         dcc.Markdown('''
       **Now apply machine learning Models**: 
       .

     ''',id='Markdown1'),
       ],className='dataBox'),

         html.Div([
            
            
            dcc.RadioItems(id = 'radio_value',
                           
                           options = [
                                      {'label': 'SVM Model', 'value': 'SVM'},
                                      {'label': 'KNN Model','value':'KNN'},
                                      {'label': 'Decision Tree','value':'Decision'},
                                      {'label': 'Random Forest','value':'Random'},
                                      {'label': 'Naive Bayes', 'value': 'Naive'}
                                      

                                      
                                      ],
                           value = 'SVM',
                           
                           
                           ),


                           
 
           
 
        ],className='radio1'),
    html.Div([
    html.Div([
        #html.P('Select the columns', id='Axis'),
            
        html.Div([
            
            dcc.Dropdown(
             
                id='X_Test',
                options=[{'label': i, 'value': i} for i in df],
                placeholder='X-Test',
                
                multi = True
                
            )
            
        ],className='xTest_value',
        ),

        html.Div([
            dcc.Dropdown(
                
                
                id='Target',
                options=[{'label': i, 'value': i} for i in df],
                placeholder='Target',
                
              
                
            )
            
        ],className='yTarget_value',),
     
    ]),

    ],className='inputValue'), 

      

    

        html.Div([
    
        dcc.Loading(dcc.Graph(id='model_graph' ,config={
                            'displayModeBar': True,
                        }),className='graph_figure'),
    
               
    
    ],className='model_figure_graph_box'),
    html.Div([
         dcc.Markdown('''
       **Accuracy is **: 

     ''',id='accuracy'),
      html.P(id='data'),
       ]),
   
    
])


@app.callback(
    Output('model_graph','figure'),
    Output('data','children'),
    [Input('X_Test','value'),
     Input('Target','value'),
     Input('radio_value','value')]
    )



def update_graph(X_Test,Target,radio_value):
    X = df[X_Test]
    y=df[Target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=109) # 70% training and 30% test

   
   

    if radio_value=='SVM':
        clfS = svm.SVC(kernel='linear') # Linear Kernel
        #Train the model using the training sets

        clfS.fit(X_train, y_train)
        
        y_pred1 = clfS.predict(X_test)
        con1=confusion_matrix(y_test,y_pred1)
        
       
        ac1=metrics.accuracy_score(y_test, y_pred1)*100
        acc1=float("{0:.2f}".format(ac1)) 
        fig1 = ff.create_annotated_heatmap(con1,)
        fig1.update_layout(
                        margin=dict(l=20, r=20, t=45, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        title = 'SVM CONFUSION MATRIX'
                        
                )
        
        return fig1,acc1
    
    elif radio_value=='KNN':
        
        knn = KNeighborsClassifier(n_neighbors=5)


        #Train the model using the training sets
        knn.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred2 = knn.predict(X_test)
        con2=confusion_matrix(y_test,y_pred2)
        ac2=metrics.accuracy_score(y_test, y_pred2)*100
        acc2=float("{0:.2f}".format(ac2))
        fig2 = ff.create_annotated_heatmap(con2)
        fig2.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        title = 'KNN CONFUSION MATRIX'
                        
                )
        return fig2,acc2
    
    elif radio_value == 'Decision':
        clfD = DecisionTreeClassifier(criterion="entropy", max_depth=3)

        #Train the model using the training sets
        clfD.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred3 = clfD.predict(X_test)
        con3=confusion_matrix(y_test,y_pred3)
        
        ac3=metrics.accuracy_score(y_test, y_pred3)*100
        acc3=float("{0:.2f}".format(ac3))
        fig3 = ff.create_annotated_heatmap(con3)
        fig3.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        title = "DECISION TREE CONFUSION MATRIX"
                        
                )
        return fig3,acc3

    elif radio_value=='Random':
        random= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
        random.fit(X_train, y_train) 
        y_pred4= random.predict( X_test)
        con4=confusion_matrix(y_test,y_pred4) 
        ac4=accuracy_score(y_test,y_pred4)*100
        acc4=float("{0:.2f}".format(ac4))
        fig4 = ff.create_annotated_heatmap(con4)  
        fig4.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        title = 'RANDOM FOREST CONFUSION MATRIX'
                        
                )
        return fig4,acc4  

    elif radio_value=='Naive':
        naive = GaussianNB()  
        naive.fit(X_train, y_train) 
        y_pred5 = naive.predict( X_test) 
        con5=confusion_matrix(y_test,y_pred5) 
        ac5=accuracy_score(y_test,y_pred5)*100
        acc5=float("{0:.2f}".format(ac5))
        fig5 = ff.create_annotated_heatmap(con5)  
        fig5.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgb(37, 41, 84)',
                        font_color='#fff',
                        plot_bgcolor='#252954',
                        title = 'NAIVE BAYES CONFUSION MATRIX'
                        
                )
        return fig5,acc5












        

    

