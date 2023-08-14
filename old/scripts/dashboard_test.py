import dash
from dash import dcc
from dash import html

app = dash.Dash()

# Create the dropdown menu
dropdown_menu = dcc.Dropdown(
    id="dropdown",
    options=[{"label": "Power", "value": "power"}, {"label": "Heat", "value": "heat"}],
    value=None,  # Initially, no value is selected
)

# Create the sliders
sliders = html.Div(id="sliders")

app.layout = html.Div([dropdown_menu, sliders])

# Callback to update the sliders based on the dropdown selection
@app.callback(
    dash.dependencies.Output("sliders", "children"),
    [dash.dependencies.Input("dropdown", "value")],
)
def update_sliders(value):
    if value is None:  # No value is selected
        return []
    elif value == "power":
        return html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.RangeSlider(
                    id="slider-{}".format(i),
                    min=0,
                    max=1,
                    step=0.1,
                    value=[0, 1],
                    marks={i: f"{i:.1f}" for i in range(11)},
                    vertical=True
                )
                for i in range(7)
            ],
        )
    elif value == "heat":
        return html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.RangeSlider(
                    id="slider-{}".format(i),
                    min=0,
                    max=1,
                    step=0.1,
                    value=[0, 1],
                    marks={i: f"{i:.1f}" for i in range(11)},
                    vertical=True
                )
                for i in range(4)
            ],
        )


if __name__ == "__main__":
    app.run_server(debug=True)
