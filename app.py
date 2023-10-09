import base64
import io

import pandas as pd

import matplotlib.pyplot as plt

from plotly.tools import mpl_to_plotly

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, ctx

from fitter import Fitter, get_distributions, get_common_distributions


external_stylesheets = ["style.css"]

app = dash.Dash(__name__, title="Fit Distributions")
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    [
        html.Link(rel="stylesheet", href="style.css"),
        html.H3(
            children="Distributions",
            className="main_title",
            style={
                "color": "white",
                "text-align": "center",
                "background-color": "black",
                "padding": "10px",
                "margin": "0px",
            },
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                className="upload-data",
                children=[
                    "Drop CSV File",
                    html.A(className="upload-button", children=["Select File"]),
                ],
            ),
            style={
                "width": "30%",
                "padding": "20px",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "textAlign": "center",
                "margin-top": "5%",
                "margin-left": "33%",
                "hover-background-color": "#24283b",
            },
            multiple=True,
        ),
        html.H5(id="file_name"),
        html.Div(
            children=[
                html.Div(id="options", style={"width": "50%"}),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            id="output-distributions",
                            style={
                                "margin-top": "2%",
                            },
                        ),
                        html.Div(
                            id="output-distributions-graph",
                            style={
                                "margin-top": "2%",
                            },
                        ),
                    ],
                ),
            ],
            style={"display": "flex"},
        ),
    ]
)


class Data:
    def __init__(self, data=None):
        self.data = data

    def store(self, data):
        self.data = data


feature_data = Data()


def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))

        feature_data.store(df)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return html.Div(
        children=[
            html.H5(
                filename
                + " | # of Numeric Features "
                + str(len(df.select_dtypes(exclude="object").columns))
                + "  | Total # of Distributions: "
                + str(len(get_distributions()))
            ),
            html.Hr(),
            html.Div(
                className="distributions_dropdown",
                children=[
                    html.Label("Select Distributions"),
                    dcc.Dropdown(
                        id="distributions",
                        options=get_distributions(),
                        value=get_common_distributions(),
                        multi=True,
                    ),
                    dcc.Checklist(
                        id="select-button", options=["Use All Distributions"]
                    ),
                ],
                style=dict(width="100%"),
            ),
            html.Br(),
            html.Div(
                className="selection_criteria",
                children=[
                    html.Label("Select Criteria"),
                    dcc.Dropdown(
                        id="criteria",
                        value="sumsquare_error",
                        options=["sumsquare_error", "aic", "bic"],
                    ),
                ],
                style=dict(width="100%"),
            ),
            html.Br(),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="features_dropdown",
                        children=[
                            html.Label("Select Column"),
                            html.Br(),
                            dcc.Dropdown(
                                id="features",
                                options=df.select_dtypes(exclude="object").columns,
                            ),
                        ],
                        style=dict(width="50%"),
                    ),
                    html.Div(
                        className="num_bins",
                        children=[
                            html.Label("# of Bins"),
                            html.Br(),
                            dcc.Input(id="bins", type="number", value=100, step=10),
                        ],
                        style=dict(width="50%"),
                    ),
                ],
                style={"display": "flex"},
            ),
            html.Br(),
            html.Button(
                "Fit Distributions", id="fit-distributions", className="btn", n_clicks=0
            ),
            html.Br(),
        ],
        style={
            "width": "50%",
            "margin-top": "2%",
            "margin-left": "40%",
            "margin-right": "10%",
        },
    )


@app.callback(
    Output("options", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)
        ]
        return children


@app.callback(
    Output("output-distributions", "children"),
    Input("fit-distributions", "n_clicks"),
    Input("distributions", "value"),
    Input("criteria", "value"),
    Input("features", "value"),
    Input("bins", "value"),
)
def display_distributions(
    n_clicks, distributions_value, criteria_value, features_value, bins_value
):
    if not n_clicks:
        return dash.no_update
    else:
        f = Fitter(
            feature_data.data[features_value].values,
            distributions=distributions_value,
            bins=bins_value,
        )
        fig, ax = plt.subplots()
        f.fit()

        summary = f.summary().reset_index()
        summary.rename(columns={"index": "Distribution(s)"}, inplace=True)
        fit_summary = dash_table.DataTable(
            summary.to_dict("records"),
            [
                {"name": i, "id": i}
                for i in summary[["Distribution(s)", "sumsquare_error", "aic", "bic"]]
            ],
        )

        best = f.get_best(method=criteria_value)
        best = pd.DataFrame(best).T.reset_index()
        best.rename(columns={"index": "Distribution"}, inplace=True)
        best_summary = dash_table.DataTable(
            best.to_dict("records"), [{"name": i, "id": i} for i in best]
        )

        return html.Div(
            children=[
                html.H5(
                    f"Best Distribution Selected by {criteria_value} for {features_value}"
                ),
                html.Hr(),
                html.Div(children=[best_summary]),
                html.Hr(),
                html.Div(
                    children=[
                        fit_summary,
                    ]
                ),
                dcc.Graph(id="distributions-graph", figure=mpl_to_plotly(fig)),
            ],
        )


if __name__ == "__main__":
    app.run_server(debug=False)
