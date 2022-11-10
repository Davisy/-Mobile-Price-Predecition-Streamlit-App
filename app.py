# import packages
import streamlit as st
import pandas as pd
import numpy as np
from os.path import dirname, join, realpath
import joblib

# add banner image
st.header("Mobile Price Prediction")
st.image("images/phones.jpg")
st.subheader(
    """
A simple machine learning app to  classify mobile price range
"""
)

# form to collect mobile phone details
my_form = st.form(key="mobile_form")


@st.cache
# function to transform Yes and No options
def func(value):
    if value == 1:
        return "Yes"
    else:
        return "No"


battery_power = my_form.number_input(
    "Total energy a battery can store in one time measured in mAh", min_value=500
)
blue = my_form.selectbox("Has bluetooth or not", (0, 1), format_func=func)

clock_speed = my_form.number_input(
    "speed at which microprocessor executes instructions", min_value=1
)

dual_sim = my_form.selectbox("Has dual sim support or not", (0, 1), format_func=func)

fc = my_form.number_input(
    "Front Camera mega pixels", min_value=0
)

four_g = my_form.selectbox("Has 4G or not", (0, 1), format_func=func)

int_memory = my_form.number_input(
    "Internal Memory in Gigabytes", min_value=2
)

m_dep = my_form.number_input(
    "Mobile Depth in cm", min_value=0
)

mobile_wt = my_form.number_input(
    "Weight of mobile phone", min_value=80
)

n_cores = my_form.number_input(
    "Number of cores of processor", min_value=1
)
pc = my_form.number_input(
    "Primary Camera mega pixels", min_value=0
)

px_height = my_form.number_input(
    "Pixel Resolution Height", min_value=0
)

px_width = my_form.number_input(
    "Pixel Resolution Width", min_value=0
)

ram = my_form.number_input(
    "Random Access Memory in Mega Bytes", min_value=256
)

sc_h = my_form.number_input(
    "Screen Height of mobile in cm", min_value=5
)

sc_w = my_form.number_input(
    "Screen Width of mobile in cm", min_value=0
)

talk_time = my_form.number_input(
    "longest time that a single battery charge will last when you are", min_value=2
)

three_g = my_form.selectbox("Has 3G or not", (0, 1), format_func=func)

touch_screen = my_form.selectbox("Has touch screen or not", (0, 1), format_func=func)

wifi = my_form.selectbox("Has wifi or not", (0, 1), format_func=func)

submit = my_form.form_submit_button(label="make prediction")

# load the model and scaler
with open(
        join(dirname(realpath(__file__)), "model/lg_classifier.pkl"),
        "rb",
) as f:
    model = joblib.load(f)

with open(join(dirname(realpath(__file__)), "model/mobile_price_scaler.pkl"), "rb") as f:
    scaler = joblib.load(f)

# result dictionary
result_dict = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost",
}
if submit:
    # collect inputs
    input = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
    }

    # create a dataframe
    data = pd.DataFrame(input, index=[0])

    # transform input
    data_scaled = scaler.transform(data)

    # perform prediction
    prediction = model.predict(data_scaled)
    output = int(prediction[0])

    # Display results of the Mobile Price prediction
    st.header("Results")
    st.write(" Price range is {} ".format(result_dict[output]))

url = "https://twitter.com/Davis_McDavid"
st.write("Developed with ❤️ by [Davis David](%s)" % url)
