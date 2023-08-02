import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_uploader as du
import uuid
import os
import ast
import torch
import torch.nn as nn
import base64
import random
import time
import io
import matplotlib.pyplot as plt

from assistants import unzip, remove_allfile_fromdir, parse_boolean
from config import create_config, create_config_single_model
from testing import webtool, predict
from plots import conf_matrix, metrics_bar, roc_auc, falsebar
from spec import plot_4s_strain, plot_processed_spectrogram, find_gps


# MNIST Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convl = nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense = nn.Sequential(
        torch.nn.Linear(14*14*128, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(1024,10))

    def forward(self, input):
        input = self.convl(input)
        input = input.view(-1, 14*14*128)
        input = self.dense(input)
        return input


def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

glitch_class_list = ['1080Lines', '1400Ripples', 'Air_Compressor', 'Blip', 'Chirp', 'Extremely_Loud', 'Helix', 'Koi_Fish', 'Light_Modulation', 'Low_Frequency_Burst', 'Low_Frequency_Lines', 'No_Glitch', 'Paired_Doves', 'Power_Line', 'Repeating_Blips', 'Scattered_Light', 'Scratchy', 'Tomte', 'Violin_Mode', 'Wandering_Line', 'Whistle']




if __name__ == '__main__':

    # Web app

    # cache = diskcache.Cache("./cache")
    # long_callback_manager = DiskcacheLongCallbackManager(cache)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)

    du.configure_upload(app, folder="files")
    test_set_folder = "files/"
    test_set_path = ""
    model_folder = "files/"
    result_text_folder = "files/"
    number_of_models = 0
    cfg = {}
    cfg_single = {}
    create_config_single_model(cfg_single, model_path=os.path.join("model", "inception-v3_checkpoint.pt"))
    wt = {}
    result_info = {}
    str_result_info = ""
    strain_path = ""
    imgsrc = ""
    gps = 0

    # Initialize elements for result page
    preview_tab, info_table, summary_table, model_options, class_options, model_result_section_list, false_section_list = ([] for i in range(7))


    running_page = html.Div([
        html.Div([
            dbc.Button("TESTING", id="test_btn"),
            html.Div([
                dcc.Loading(id="test_loading", children=[html.Div(id="wtpreview")], type="circle")
            ]),
            dbc.Button("RESULT", id="result_btn", disabled=True),
            html.Div(id="link_download_wt")
        ], className="text-center")
        # dbc.Col([
        #     dbc.Button("TESTING", id="test_btn"),
        #     dcc.Loading(id="test_loading", children=[html.Div(id="wtpreview")], type="circle"),
        #     dbc.Button("RESULT", id="result_btn", disabled=True),
        #     html.Div(id="link_download_wt")
        # ], lg={"size":2, "offset":5}, xl={"size":2, "offset":5}),


    ])




    initial_page_1 = html.Div([
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H4("Please upload your test set as a zip file",
                    style={'textAlign': 'center'}),
                dbc.Container(
                    du.Upload(
                        id="test_set_uploader",
                        text="Drag or Click",
                        upload_id=uuid.uuid1(),
                        filetypes=["zip"]
                    )
                ),
                html.Br(),
                dbc.Col([
                    html.H5("Uploaded files:"),
                    html.Ul(id="test_set_list", children=[]),
                    dbc.Button("DELETE", id="delete_test_set_btn"),
                ], width=10)
            ], xl=6, lg=6, md=6, sm=12, xs=12),

            dbc.Col([
                html.H4("Please upload your PyTorch models",
                    style={'textAlign': 'center'}),
                dbc.Container(
                    du.Upload(
                        id="model_uploader",
                        text="Drag or Click",
                        upload_id=uuid.uuid1(),
                        filetypes=["pt", "pth"]
                    )
                ),
                html.Br(),
                dbc.Col([
                    html.H5("Uploaded files:"),
                    html.Ul(id="model_list", children=[]),
                    dbc.Button("DELETE", id="delete_model_btn"),
                ], width=10)

            ], xl=6, lg=6, md=6, sm=12, xs=12)
        ], justify="center"),
        html.Hr(),
        dbc.Col([
            html.H4("Or upload your previous test result text file:", style={"align": "center"}),
            du.Upload(
                id="result_uploader",
                text="Drag or Click",
                upload_id=uuid.uuid1(),
                filetypes=["txt"]
            ),
            html.Div([
                html.Br(),
                dbc.Button("RESULT", id="text_to_result_btn", disabled=True, href="/result")
            ], className="text-center"),

        ], md={"size":8, "offset":2}, lg={"size":6, "offset":3}, xl={"size":6, "offset":3}),

        html.Hr(),
        html.H4("Custom Configuration", style={'textAlign': 'center'}),
        dbc.Col([
            dbc.Table([
                html.Tr([
                    html.Th("Hyper-Parameters"),
                    html.Th("Value", style={"text-align": "center"})
                ]),
                html.Tr([
                    html.Td("Device"),
                    html.Td([
                        dbc.RadioItems(
                            id="device_select",
                            options=[
                            {'label': 'CPU', 'value': 'CPU'},
                            {'label': 'Cuda', 'value': 'Cuda', 'disabled': True}
                            ],
                            value="CPU",
                            inline=True
                        )
                    ], style={"text-align": "center"})
                ]),
                html.Tr([
                    html.Td("Batch size"),
                    html.Td([
                        dbc.RadioItems(
                            id="batchsize_select",
                            options=[
                            {'label': '16', 'value': 16},
                            {'label': '8', 'value': 8},
                            {'label': '4', 'value': 4},
                            ], value=8,
                            inline=True
                        )
                    ], style={"text-align": "center"})
                ]),
                html.Tr([
                    html.Td("Number of workers"),
                    html.Td([
                        dbc.RadioItems(
                            id="numworkers_select",
                            options=[
                            {'label': "4", 'value': 4},
                            {'label': "8", 'value': 8, 'disabled': True}
                            ],
                            value=4,
                            inline=True
                        )
                    ], style={"text-align": "center"})
                ]),
                html.Tr([
                    html.Td("Shuffle"),
                    html.Td([
                        dbc.RadioItems(
                            id="shuffle_select",
                            options=[
                            {'label': 'True', 'value': "True", "disabled": True},
                            {'label': 'False', 'value': "False"}
                            ],
                            value="False",
                            inline=True
                        )
                    ], style={"text-align": "center"})
                ])
            ], hover=True, striped=True),
        ], md={"size":8, "offset":2}, lg={"size":6, "offset":3}, xl={"size":6, "offset":3}),
        dbc.Row([
            dbc.Button("SUBMIT", id="submit_btn", size="lg")
        ], justify="center"),
        html.Br(),
        dbc.Container(id="config_info"),
        html.Div(children=[], id="running_section"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Please press the button below to start testing."), close_button=False),
            dbc.ModalBody(children=[running_page], id="running_modal_body"),
            dbc.ModalFooter("Thank you for using this tool.")
        ], id="running_modal", keyboard=False, backdrop="static")
    ])

    initial_page_2 = html.Div([
        html.Br(),
        dbc.Col([
            # html.H4("1. Please select pre-trained PyTorch model",
            #         style={'textAlign': 'center'}),
            # dbc.Container(
            #     du.Upload(
            #         id="model_uploader_2",
            #         text="Drag or Click",
            #         upload_id=uuid.uuid1(),
            #         filetypes=["pt", "pth"],
            #         max_files=1,
            #     )
            # ),
            # html.Div(style={'textAlign': 'center'}, id="model_uploader_2_msg"),
            # html.Hr(),
            html.H4("1. Please upload a .hdf5 file of strain data from GWOSC",
                    style={"textAlign": "center"}),
            html.Div([
                html.Br(),
                html.P("Hint: You can download the public strain data from:"),
                html.A("Gravitational Wave Open Science Center", href="https://www.gw-openscience.org/data/",
                    target="_blank"),
            ], className="text-center"),
            dbc.Container(
                du.Upload(
                    id="strain_uploader",
                    text="Drag or Click",
                    upload_id=uuid.uuid1(),
                    filetypes=["hdf5"],
                    max_files=1,
                    disabled=False,
                ),
            ),
            html.P(style={'textAlign': 'center'}, id="strain_uploader_msg"),
            html.Hr(),
            html.H4("2. Select your interested time of the strain data",
                    style={"textAlign": "center"}),
            html.Div([
                html.P("Select a time (from the strain file GPS time)"),
                dbc.Row([
                    dbc.Col([
                        html.Button("-0.1", id="minus_gps")
                    ]),
                    dbc.Col([
                        html.P(id="gps_time")
                    ]),
                    dbc.Col([
                        html.Button("+0.1", id="plus_gps")
                    ], className="text-center")
                ]),
                dcc.Slider(min=2, max=4094, step=0.1, included=False, marks=None, value=2048,
                           tooltip={"placement": "bottom", "always_visible": True},
                           id="strain_time_slider"),
                dbc.Button("SUBMIT", id="strain_submit", disabled=True),
                html.Div(id="strain_output")
            ], className="text-center"),
            html.Hr(),
            html.H4("3. Data preview", style={"textAlign": "center"}),
            html.Div([
                dbc.Col([
                    html.Img(id="strain_4s", src=""),
                ]),
            ], className="text-center"),
            html.Hr(),
            html.H4("4. Prediction", style={"textAlign": "center"}),
            html.Div([
                html.P("Multi-duration Q-Transformed"),
                html.Img(id="transformed_strain", src="")
            ], className="text-center"),
            html.Div(id="strain_prediction", className="text-center"),
        ])
    ])


    initial_page = html.Div([
        html.H1("Deep Learning Visualization Web Tool", style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col([
                dcc.Tabs(children=[
                    dcc.Tab([initial_page_1], label="Model Testing"),
                    dcc.Tab([initial_page_2], label="Glitch Prediction")
                ])
            ], width={"size": 10, "offset": 1})
        ])
    ])




    def generate_result_elements(result_info):
        # Preview tab
        preview_rows_per_three = (len(result_info["info"][0]) // 3) + (len(result_info["info"][0]) % 3)
        preview_tab = [html.Br()]
        count = 0
        for i in range(preview_rows_per_three):
            cols = []
            for j in range(3):
                if count < len(result_info["info"][0]):
                    pathname = os.path.join(result_info["test_set_path"], result_info["info"][0][count])
                    filename = random.choice(os.listdir(pathname))
                    filepath = os.path.join(pathname, filename)
                    cols.append(dbc.Col([
                        html.H4(result_info["info"][0][count]),
                        html.Img(src=b64_image(filepath), style={'height': '80%', 'width': '80%'})
                    ]))
                    count += 1
                else:
                    cols.append(dbc.Col([]))
            row = dbc.Row(cols)
            preview_tab.append(row)

        # Info table
        info_table_elements = [
            html.Tr([
                html.Td(i),
                html.Td(j),
                html.Td(k)
            ]) for i, j, k in zip(result_info["info"][0], result_info["info"][1], result_info["info"][2])
        ]

        info_table_body = []

        for i in info_table_elements:
            info_table_body.append(i)

        info_table_body.append(
            html.Tr([
                html.Td("Overall"),
                html.Td(sum(result_info["info"][1])),
                html.Td("100.00%")
            ])
        )

        info_table = [
            html.Thead([
                html.Tr([
                    html.Th("Class Name"),
                    html.Th("Count"),
                    html.Th("Percentage")
                ])
            ]),
            html.Tbody(info_table_body)
        ]

        # Summary table
        summary_table_elements = []
        for (index, (key, value)) in enumerate(result_info["models"].items()):
            summary_table_elements.append(
                html.Tr([
                    html.Td(key),
                    html.Td(str(int(sum(result_info["result"][index][1]))) + "/" + str(
                        int(sum(result_info["result"][index][3])))),
                    html.Td(format(100 * (sum(result_info["result"][index][1]) / sum(result_info["result"][index][3])),
                                   ".3f") + "%"),
                    html.Td(format(100 * (sum(result_info["result"][index][2]) / sum(result_info["result"][index][3])),
                                   ".3f") + "%"),
                    html.Td("Wrong class")
                ])
            )

        summary_table = [
            html.Thead([
                html.Tr([
                    html.Th("Model"),
                    html.Th("Correct"),
                    html.Th("Top-1 Accuracy"),
                    html.Th("Top-5 Accuracy"),
                    html.Th("Top Incorrect Predictions")
                ])
            ]),
            html.Tbody(summary_table_elements)
        ]

        # Drop down menu items of models
        model_options = []
        for i in result_info["models"]:
            model_dict = {"label": i, "value": i}
            model_options.append(model_dict)

        # Drop down menu items of classes
        class_options = []
        for i in result_info["info"][0]:
            class_dict = {"label": i, "value": i}
            class_options.append(class_dict)
        class_options.append({"label": "All", "value": "All"})
        class_options.append({"label": "Average", "value": "Average"})

        # Model result section
        model_result_section_list = []
        for (index, (key, value)) in enumerate(result_info["models"].items()):

            accuracy_table_body = []
            for j in range(len(result_info["result"][index][0])):
                accuracy_table_body.append(
                    html.Tr([
                        html.Td(result_info["result"][index][0][j]),
                        html.Td(str(int(result_info["result"][index][1][j])) + "/" + str(
                            int(result_info["result"][index][3][j]))),
                        html.Td(format(100 * (result_info["result"][index][1][j] / result_info["result"][index][3][j]),
                                       ".3f") + "%"),
                        html.Td(format(100 * (result_info["result"][index][2][j] / result_info["result"][index][3][j]),
                                       ".3f") + "%")
                    ])
                )

            accuracy_table_body.append(
                html.Tr([
                    html.Td("Overall"),
                    html.Td(str(int(sum(result_info["result"][index][1]))) + "/" + str(
                        int(sum(result_info["result"][index][3])))),
                    html.Td(format(100 * (sum(result_info["result"][index][1]) / sum(result_info["result"][index][3])),
                                   ".3f") + "%"),
                    html.Td(format(100 * (sum(result_info["result"][index][2]) / sum(result_info["result"][index][3])),
                                   ".3f") + "%"),
                ])
            )

            accuracy_table = [
                html.Thead([
                    html.Tr([
                        html.Th("Class Name"),
                        html.Th("Correct"),
                        html.Th("Top-1 Accuracy"),
                        html.Th("Top-5 Accuracy")
                    ])
                ]),
                html.Tbody(accuracy_table_body)
            ]

            # The section begin here
            model_result_section = html.Div([
                html.Br(),
                html.H3("Accuracy"),
                dbc.Table(accuracy_table, hover=True, responsive=True, striped=True),
                html.Hr(),
                html.Br(),
                html.H3("Confusion Matrix"),
                html.Div([
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                children=[
                                    dcc.Graph(
                                        figure=conf_matrix(result_info["test_tensors_list"][index],
                                                           result_info["pred_tensors_list"][index],
                                                           result_info["info"][0]), style={"height": 700})
                                ],
                                title="Without normalization"
                            ),
                            dbc.AccordionItem(
                                children=[
                                    dcc.Graph(
                                        figure=conf_matrix(result_info["test_tensors_list"][index],
                                                           result_info["pred_tensors_list"][index],
                                                           result_info["info"][0],
                                                           norm_true=True), style={"height": 700})
                                ],
                                title="Normalized over true labels"
                            ),
                            dbc.AccordionItem(
                                children=[
                                    dcc.Graph(
                                        figure=conf_matrix(result_info["test_tensors_list"][index],
                                                           result_info["pred_tensors_list"][index],
                                                           result_info["info"][0],
                                                           norm_predict=True), style={"height": 700})
                                ],
                                title="Normalized over predicted labels"
                            )
                        ],
                    )
                ]),
                # dcc.Graph(
                #     figure=conf_matrix(result_info["test_tensors_list"][index], result_info["pred_tensors_list"][index],
                #                        result_info["info"][0]), style={"height": 700}),
                html.Hr(),
                html.Br(),
                html.H3("Metrics"),
                dcc.Graph(
                    figure=metrics_bar(result_info["test_tensors_list"][index], result_info["pred_tensors_list"][index],
                                       result_info["info"][0])),
                html.Hr(),
                html.Br()
            ], id=key + "_result_section")

            model_result_section_list.append(model_result_section)

        # Model false prediction section
        false_section_list = []
        for (index, (key, value)) in enumerate(result_info["models"].items()):
            rows_per_four = (len(result_info["falseimgs_all_model"][index]) // 4) + (
                        len(result_info["falseimgs_all_model"][index]) % 4)
            false_tab = [html.Br()]
            count = 0
            for i in range(rows_per_four):
                cols = []
                for j in range(4):
                    if count < len(result_info["falseimgs_all_model"][index]):
                        path = result_info["falseimgs_all_model"][index][count]["image_path"]
                        prediction_class = result_info["info"][0][
                            (result_info["falseimgs_all_model"][index][count]["prediction"])]
                        true_class = result_info["info"][0][(result_info["falseimgs_all_model"][index][count]["label"])]
                        cols.append(dbc.Col([
                            dbc.Card([
                                dbc.CardImg(src=b64_image(path), top=True),
                                dbc.CardBody([
                                    html.P(["Prediction: ", html.Strong(prediction_class)]),
                                    html.P(["True label: ", html.Strong(true_class)])
                                ])
                            ], color="secondary", outline=True)
                        ]))
                        count += 1
                    else:
                        cols.append(dbc.Col([]))
                row = dbc.Row(cols)
                false_tab.append(row)
            false_section_list.append(false_tab)

        return preview_tab, info_table, summary_table, model_options, class_options, model_result_section_list, false_section_list


    def update_result_layout(preview_tab, info_table, summary_table, model_options, class_options, model_result_section_list, false_section_list):
        layout = html.Div([
            html.H1("Deep Learning Visualization Web Tool", style={'textAlign': 'center'}),
            dbc.Row(
                dbc.Col(
                    dcc.Tabs(id="result_tab", value="distri", children=[
                        dcc.Tab(preview_tab, label="Test Set Preview"),
                        dcc.Tab([
                            html.Br(),
                            html.H3("Test set distribution"),
                            dbc.Table(info_table, hover=True, responsive=True, striped=True),
                            html.Hr(),
                            html.Br(),
                            html.H3("Distribution Graph"),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Histogram", "value": "Histogram"},
                                    {"label": "Pie Chart", "value": "Pie Chart"}
                                ],
                                id="distri_select", value="Histogram"
                            ),
                            dcc.Graph(id="distribution")
                        ], label="Test Set Information", value="distri"),
                        dcc.Tab([
                            html.Br(),
                            html.H3("Models Performances Summary"),
                            dbc.Table(summary_table, hover=True, responsive=True, striped=True),
                            html.Hr(),
                            html.Br(),
                            html.H3("Receiver Operating Characteristic Curve"),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.P("Model"),
                                    dcc.Dropdown(
                                        id="ROC_model_dropdown_1",
                                        options=model_options
                                    ),
                                    html.Br(),
                                    html.P("Class"),
                                    dcc.Dropdown(
                                        id="ROC_class_dropdown_1",
                                        options=class_options
                                    ),
                                    html.Br(),
                                    dbc.Button("Generate", id="ROC_btn_1"),
                                    html.Div(id="ROC_1", children=[])
                                ]),
                                dbc.Col([
                                    html.P("Model"),
                                    dcc.Dropdown(
                                        id="ROC_model_dropdown_2",
                                        options=model_options
                                    ),
                                    html.Br(),
                                    html.P("Class"),
                                    dcc.Dropdown(
                                        id="ROC_class_dropdown_2",
                                        options=class_options
                                    ),
                                    html.Br(),
                                    dbc.Button("Generate", id="ROC_btn_2"),
                                    html.Div(id="ROC_2", children=[])
                                ])
                            ], justify="center", id="ROC_row"),
                            html.Hr(),
                            html.Br(),
                            html.H3("False Prediction Bar Chart"),
                            dcc.Graph(figure=falsebar(result_info["falseimgs_all_model"], result_info["models"], result_info["info"][0]))
                        ], label="Model Comparison"),
                        dcc.Tab([
                            html.Br(),
                            dcc.Dropdown(
                                id="model_dropdown",
                                options=model_options,
                                placeholder="Please select a model",
                            ),
                            html.Div(
                                children=[],
                                id="model_result_section"
                            )
                        ], label="Single Model Performance"),
                        dcc.Tab([
                            html.Br(),
                            dcc.Dropdown(
                                id="false_dropdown",
                                options=model_options,
                                placeholder="Please select a model",
                            ),
                            html.Div(
                                children=[],
                                id="false_section"
                            )
                        ], label="False Predictions")
                    ]),
                    width={"size": 10, "offset": 1}
                )
            )
        ])

        return layout



    result_layout = html.Div([html.H1("No result yet!")])
    result_page = html.Div([])


    def update_result_page(str_result_info, result_page):

        global model_result_section_list
        global false_section_list

        if str_result_info == "":
            result_info = {}
            result_layout = html.Div([html.H1("No result yet!")])
            result_page = result_layout
            return result_page, result_info
        else:
            result_layout = html.Div([html.H1("No result yet!")])
            result_info = ast.literal_eval(str_result_info)
            preview_tab, info_table, summary_table, model_options, class_options, model_result_section_list, false_section_list = generate_result_elements(result_info)
            result_page = update_result_layout(preview_tab, info_table, summary_table, model_options, class_options, model_result_section_list, false_section_list)
            return result_page, result_info


    app.validation_layout = html.Div([
        initial_page,
        running_page,
        result_page
    ])


    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div([initial_page], id="page-content")
    ])






    # Upload test set
    @app.callback(Output("test_set_list", "children"),
                  Input("test_set_uploader", "isCompleted"),
                  Input("delete_test_set_btn", "n_clicks"),
                  State("test_set_uploader", "upload_id"),
                  State("test_set_uploader", "fileNames"))
    def handle_upload_testset(isCompleted, n_clicks, upload_id, fileNames):
        ctx = dash.callback_context
        global test_set_folder
        global test_set_path
        if "test_set_uploader" in ctx.triggered[0]["prop_id"]:
            if test_set_folder == "files/":
                test_set_folder += str(upload_id)
            # unzip
            zip_path = os.path.join(test_set_folder, fileNames[0])
            unzip(zip_path)
            test_set_path = zip_path[:-4]
            display = []
            for i in os.listdir(test_set_folder):
                if i == "__MACOSX" or i == ".DS_Store":
                    pass
                else:
                    display.append(html.Li(i))
            return display
        elif "delete_test_set_btn" in ctx.triggered[0]["prop_id"]:
            if test_set_folder != "files/":
                remove_allfile_fromdir(test_set_folder)
                return [html.Li("Your test set has been deleted.")]
            else:
                return [html.Li("No files yet.")]
        else:
            return []




    # Upload models
    @app.callback(Output("model_list", "children"),
                  Input("model_uploader", "isCompleted"),
                  Input("delete_model_btn", "n_clicks"),
                  State("model_uploader", "upload_id"),
                  State("model_uploader", "fileNames"))
    def handle_upload_model(isCompleted, n_clicks, upload_id, fileNames):
        ctx = dash.callback_context
        global model_folder
        if "model_uploader" in ctx.triggered[0]["prop_id"]:
            if model_folder == "files/":
                model_folder += str(upload_id)
            return [
                html.Li(i) for i in os.listdir(model_folder)
            ]
        elif "delete_model_btn" in ctx.triggered[0]["prop_id"]:
            if model_folder != "files/":
                remove_allfile_fromdir(model_folder)
                return [html.Li("All model files have been deleted.")]
            else:
                return [html.Li("No files yet.")]
        else:
            return []



    # Create Configuration
    @app.callback([Output("config_info", "children"),
                   Output("submit_btn", "disabled"),
                   Output("running_modal", "is_open")
                   # Output("running_section", "children")
                   # Output("turn_to_running", "disabled")
                   ],
                  Input("submit_btn", "n_clicks"),
                  State("batchsize_select", "value"),
                  State("numworkers_select", "value"),
                  State("shuffle_select", "value")
                  )
    def submit_config(n_clicks, batch_size, num_workers, shuffle):
        global test_set_folder
        global model_folder
        global test_set_path
        global number_of_models
        global cfg
        models = {}
        shuffle_bool = parse_boolean(shuffle)
        if not n_clicks:
            raise PreventUpdate
        if (test_set_folder == "files/" or model_folder == "files/") or\
                ((len(os.listdir(test_set_folder))) == 0 or (len(os.listdir(model_folder)) == 0)):
            return html.P("Upload unfinished."), False, False
        else:
            number_of_models = len(os.listdir(model_folder))
            for i in os.listdir(model_folder):
                if i.endswith(".pt"):
                    name = i[:-3]
                else:
                    name = i[:-4]
                models[name] = os.path.join(model_folder, i)
            cfg = create_config(cfg, batch_size, num_workers, test_set_path, shuffle_bool, number_of_models, models)
            return [
                html.P("Submitted successfully.")
            ], True, True





    @app.callback([Output("wtpreview", "children"),
                   Output("result_btn", "disabled"),
                   Output("test_btn", "disabled"),
                   Output("link_download_wt", "children")],
                  Input("test_btn", "n_clicks"))
    def test_begin(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        global wt
        global cfg
        global result_info
        global str_result_info

        start_time = time.process_time()
        wt = webtool(wt, cfg)

        # The dictionary that will be converted to string for user repeat usage
        result_info["test_set_path"] = cfg["test_set_path"]
        result_info["models"] = cfg["models"]
        result_info["info"] = wt["info"]
        result_info["result"] = wt["result"]
        test_tensors_list = [i.tolist() for i in wt["test_tensors"]]
        result_info["test_tensors_list"] = test_tensors_list
        pred_tensors_list = [i.tolist() for i in wt["pred_tensors"]]
        result_info["pred_tensors_list"] = pred_tensors_list
        pred_score_arrays_list = [i.tolist() for i in wt["pred_score_arrays"]]
        result_info["pred_score_arrays_list"] = pred_score_arrays_list
        result_info["falseimgs_all_model"] = wt["falseimgs_all_model"]
        str_result_info = str(result_info)

        end_time = time.process_time()
        processed = "Processed time: {:.2f} s".format((end_time-start_time)*10)

        return html.Div([
            html.P("Finished."),
            html.P(processed)
        ]), False, True, [dbc.Button("Download .txt file", id="download_wt_btn", color="link"),
                          html.P("Next time, you can generate the result page immediately after uploaded this text file."),
                          dcc.Download(id="download_wt")]



    @app.callback(Output("download_wt", "data"),
                  Input("download_wt_btn", "n_clicks"))
    def download_wt_text(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        global str_result_info
        return dict(content=str_result_info, filename="result_text.txt")






    # Initial page: upload text file region
    @app.callback(Output("text_to_result_btn", "disabled"),
                   Input("result_uploader", "isCompleted"),
                   State("result_uploader", "upload_id"),
                  State("result_uploader", "fileNames"))
    def handle_upload_result(isCompleted, upload_id, fileNames):
        global str_result_info
        global result_info
        if isCompleted:
            text_path = "files/" + str(upload_id) + "/" + fileNames[0]
            with open(text_path) as text_file:
                data = text_file.read()
            str_result_info = data
            result_info = ast.literal_eval(str_result_info)
            return False
        return dash.no_update



    # Turning to result page
    @app.callback(Output("url", "pathname"),
                  Input("result_btn", "n_clicks"))
    def turning_to_result_page(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return "/result"




    # Handle page exchanges
    @app.callback(Output("page-content", "children"),
                  Input("url", "pathname"))
    def display_page(pathname):
        if pathname == "/result":
            global result_page
            global result_info
            global str_result_info
            result_page, result_info = update_result_page(str_result_info, result_page)
            return result_page
        else:
            return initial_page



    # Prediction callbacks

    # # handle model upload
    # @app.callback([Output("model_uploader_2_msg", "children"),
    #                Output("strain_uploader", "disabled")],
    #               Input("model_uploader_2", "isCompleted"),
    #               State("model_uploader_2", "upload_id"),
    #               State("model_uploader_2", "fileNames"))
    # def single_model_upload(isCompleted, upload_id, fileNames):
    #     global cfg_single
    #     if isCompleted:
    #         model_path = os.path.join("files", str(upload_id), fileNames[0])
    #         create_config_single_model(cfg_single, model_path=model_path)
    #         return [], False
    #     return dash.no_update, True

    # handle strain upload
    @app.callback(Output("strain_submit", "disabled"),
                  Input("strain_uploader", "isCompleted"),
                  State("strain_uploader", "upload_id"),
                  State("strain_uploader", "fileNames"))
    def strain_upload(isCompleted, upload_id, fileNames):
        global strain_path
        global gps
        if isCompleted:
            strain_path = os.path.join("files", str(upload_id), fileNames[0])
            gps = find_gps(strain_path)
            return False
        return dash.no_update

    # handle submit, plot the strains, and predicting
    @app.callback([Output("strain_4s", "src"),
                   Output("transformed_strain", "src"),
                   Output("strain_prediction", "children"),],
                  Input("strain_submit", "n_clicks"),
                  State("strain_time_slider", "value"))
    def plot_td_qt(n_clicks, value):
        if not n_clicks:
            raise PreventUpdate
        global cfg_single
        global strain_path
        global imgsrc
        strain = plot_4s_strain(hdf5=strain_path, start=value-2, end=value+2)
        imgsrc = plot_processed_spectrogram(hdf5=strain_path, start=value-2, end=value+2)
        buf = plot_processed_spectrogram(hdf5=strain_path, start=value-2, end=value+2, to_predict=True)
        prediction, prob_1, prob_2 = predict(cfg_single=cfg_single, imgsrc=buf, class_list=glitch_class_list)
        if prob_2 == None:
            msg = html.Div([
                html.P("{} ({:.3%})".format(prediction, prob_1))
            ], className="text-center")
        else:
            msg = html.Div([
                html.P("{} ({:.3%})".format(prediction, prob_1)),
                html.P("{} ({:.3%})".format(prob_2[1], prob_2[0]))
            ], className="text-center")
        return strain, imgsrc, msg

    # showing gps time
    @app.callback(Output("gps_time", "children"),
                  Input("strain_time_slider", "value"))
    def show_gps_time(value):
        global gps
        if gps == 0:
            return "GPS: -"
        else:
            return "GPS: {}".format(gps+value)

    # control time by button
    @app.callback(Output("strain_time_slider", "value"),
                  [Input("minus_gps", "n_clicks"),
                   Input("plus_gps", "n_clicks"),
                   State("strain_time_slider", "value")])
    def plus_minus_time(n_clicks1, n_clicks2, value):
        ctx = dash.callback_context
        if ctx.triggered[0]['prop_id'] == "minus_gps.n_clicks":
            if value-0.1 >= 2:
                return value-0.1
            else:
                return value
        elif ctx.triggered[0]['prop_id'] == "plus_gps.n_clicks":
            if value+0.1 <= 4094:
                return value+0.1
            else:
                return value
        else:
            return dash.no_update


















    # Result page callbacks
    @app.callback(Output("distribution", "figure"),
                  Input("distri_select", "value"))
    def plot_distribution(value):
        if value == "Histogram":
            fig = px.histogram(y=result_info["info"][0], x=result_info["info"][1], text_auto=True)
            fig.layout.yaxis.title = "Class"
            fig.layout.xaxis.title = "Count"
            fig.layout.title = "Distribution of Classes"
            fig.layout.height = 35 * len(result_info["info"][0])
            return fig
        else:
            fig = px.pie(names=result_info["info"][0], values=result_info["info"][1])
            fig.layout.title = "Distribution of Classes"
            return fig


    @app.callback(Output("model_result_section", "children"),
                  Input("model_dropdown", "value"))
    def return_result_section(value):
        for (index, (k, v)) in enumerate(result_info["models"].items()):
            if k == value:
                return model_result_section_list[index]
        return html.Div([])


    @app.callback(Output("false_section", "children"),
                  Input("false_dropdown", "value"))
    def return_false_section(value):
        for (index, (k, v)) in enumerate(result_info["models"].items()):
            if k == value:
                return false_section_list[index]
        return html.Div([])


    @app.callback(Output("ROC_1", "children"),
                  Input("ROC_btn_1", "n_clicks"),
                  State("ROC_model_dropdown_1", "value"),
                  State("ROC_class_dropdown_1", "value"))
    def plot_roc_1(n_clicks, model, class_name):
        if not n_clicks:
            raise PreventUpdate
        if class_name == "Average" or class_name == "All":
            class_idx = -1
        else:
            class_idx = result_info["info"][0].index(class_name)

        model_idx = 0
        for (index, (k, v)) in enumerate(result_info["models"].items()):
            if k == model:
                model_idx = index

        if class_name == "Average":
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=True, all=False)
        elif class_name == "All":
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=False, all=True)
        else:
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=False, all=False)

        return dcc.Graph(figure=fig)


    @app.callback(Output("ROC_2", "children"),
                  Input("ROC_btn_2", "n_clicks"),
                  State("ROC_model_dropdown_2", "value"),
                  State("ROC_class_dropdown_2", "value"))
    def plot_roc_2(n_clicks, model, class_name):
        if not n_clicks:
            raise PreventUpdate
        if class_name == "Average" or class_name == "All":
            class_idx = -1
        else:
            class_idx = result_info["info"][0].index(class_name)

        model_idx = 0
        for (index, (k, v)) in enumerate(result_info["models"].items()):
            if k == model:
                model_idx = index

        if class_name == "Average":
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=True, all=False)
        elif class_name == "All":
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=False, all=True)
        else:
            fig = roc_auc(result_info["test_tensors_list"][model_idx], result_info["pred_score_arrays_list"][model_idx],
                          result_info["info"][0], class_idx=class_idx, average=False, all=False)

        return dcc.Graph(figure=fig)


    # if __name__ == '__main__':
    app.run_server(debug=True)