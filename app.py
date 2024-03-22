import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import base64


# Set the page configuration as the very first command
st.set_page_config(page_title="Beyond The Haze", layout="wide")


# Load models from files
def load_models():
    with open('random_rorest_regressor_model_aqi.pickle', 'rb') as file:
        rf_regressor = pickle.load(file)
    with open('decision_tree_classifier_model_time.pickle', 'rb') as file:
        dt_classifier = pickle.load(file)
    return rf_regressor, dt_classifier

# Preprocess inputs for AQI prediction
def preprocess_input_aqi(pm25, pm10, no, no2, nox, co, so2):
    return pd.DataFrame({
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO': [no],
        'NO2': [no2],
        'NOx': [nox],
        'CO': [co],
        'SO2': [so2]
    })
    
# Function to get base64 encoding of an image file for the background
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background image from a local file
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
set_background('Smoky2.png')
    
    
# Preprocess inputs for time prediction
def preprocess_input_time(city_index, month, day):
    return pd.DataFrame({
        'City': 5*[city_index],
        'Month': 5*[month],
        'Day': 5*[day],
        'Time': [0, 1, 2, 3, 4]
    })

# Initialize session state
def initialize_state():
    if 'pm25' not in st.session_state:
        st.session_state.pm25 = 0.0
    if 'pm10' not in st.session_state:
        st.session_state.pm10 = 0.0
    if 'no' not in st.session_state:
        st.session_state.no = 0.0
    if 'no2' not in st.session_state:
        st.session_state.no2 = 0.0
    if 'nox' not in st.session_state:
        st.session_state.nox = 0.0
    if 'co' not in st.session_state:
        st.session_state.co = 0.0
    if 'so2' not in st.session_state:
        st.session_state.so2 = 0.0
    if 'city' not in st.session_state:
        st.session_state.city = "Ahmedabad"
    if 'date' not in st.session_state:
        st.session_state.date = pd.Timestamp.now().date()

# Reset callbacks
def reset_aqi_callback():
    for key in ['pm25', 'pm10', 'no', 'no2', 'nox', 'co', 'so2']:
        st.session_state[key] = 0.0

def reset_time_callback():
    st.session_state.city = "Ahmedabad"
    st.session_state.date = pd.Timestamp.now().date()



# Main application
def main():
    
    # Set page config
    
    # Correct placement for set_page_config
    # st.set_page_config(page_title="Air Quality Prediction", layout="wide")
    
    
    city_names = ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
                  'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
                  'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
                  'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai',
                  'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
                  'Visakhapatnam']
    
    # st.set_page_config(page_title="Air Quality Prediction", layout="wide")

    # Custom CSS for styling
    st.markdown("""
    <style>
    
        /* Streamlit styling */
        .css-1d391kg {
            padding: 2rem;
        }

        .st-bx {
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
        }

        .st-bx input, .st-bx select, .st-bx button {
            border: 1px solid #ced4da;
        }

        button {
            width: 100%;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Streamlit styling */
        .main .block-container {
            padding: 2rem;
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        h1, h2 {
            color: #ffffff; /* White color for title and headers */
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6); /* Text shadow for better readability */
        }
        
        
        /* Custom CSS */
        body {
            background-color: #f4f4f4;
            color: #333333;
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 5px;
        }
        
        .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
            color: #0e1117; /* Dark text color for readability */
            background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background */
            border-radius: 5px; /* Rounded corners */
            border: 1px solid #ced4da; /* Border color */
            padding: 10px; /* Inner spacing */
        }
    
        .card {
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 15px;
            border-radius: 10px;
            background-color: white;
            margin-bottom: 10px;
        }
        
        /* Success message styling */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background */
            color: #0e1117; /* Dark text color for readability */
            border-radius: 10px; /* Rounded corners */
            border: 1px solid #28a745; /* Green border color */
            padding: 20px; /* Inner spacing */
            margin: 10px 0; /* Outer spacing */
        }
    
    </style>
    """, unsafe_allow_html=True)

    st.title("🌿 Air Quality Prediction App")

    initialize_state()

    with st.sidebar:
        st.info("Enter the required details to predict Air Quality Index (AQI) and the Best Time of Day for Air Quality.")

    rf_regressor, dt_classifier = load_models()

    st.header("🌤 Predict AQI")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input('PM2.5', min_value=0.0, max_value=500.0, value=st.session_state.pm25, key='pm25')
            st.number_input('PM10', min_value=0.0, max_value=500.0, value=st.session_state.pm10, key='pm10')
        with col2:
            st.number_input('NO', min_value=0.0, max_value=1500.0, value=st.session_state.no, key='no')
            st.number_input('NO2', min_value=0.0, max_value=1500.0, value=st.session_state.no2, key='no2')
        with col3:
            st.number_input('NOx', min_value=0.0, max_value=1500.0, value=st.session_state.nox, key='nox')
            st.number_input('CO', min_value=0.0, max_value=1500.0, value=st.session_state.co, key='co')
            st.number_input('SO2', min_value=0.0, max_value=1500.0, value=st.session_state.so2, key='so2')

        def classify_aqi(aqi_value):
            aqi_ranges = {
                'Good': [13.0, 51.0],
                'Satisfactory': [51.0, 101.0],
                'Moderate': [101.0, 201.0],
                'Poor': [201.0, 301.0],
                'Very Poor': [301.0, 401.0],
                'Severe': [401.0, 999999.0]
            }
            
            descriptions = {
                'Good': "Air quality is satisfactory, and poses little or no risk to health. Enjoy outdoor activities!",
                'Satisfactory': "Air quality is acceptable; however, there may be a concern for some people who are unusually sensitive to air pollution.",
                'Moderate': "Air quality is okay; however, there may be a concern for some people who are unusually sensitive to air pollution. Avoid prolonged outdoor activities.",
                'Poor': "Air quality is poor and may cause health issues, particularly for people with respiratory or heart problems. Minimize outdoor activities.",
                'Very Poor': "Air quality is very poor and may cause serious health effects. Avoid all outdoor activities and stay indoors as much as possible.",
                'Severe': "Air quality is extremely poor and poses a severe risk to health. It's recommended to stay indoors, use air purifiers, and wear N95 masks if you must go outside."
            }

            recommendations = {
                'Good': "Enjoy outdoor activities and maintain a healthy lifestyle.",
                'Satisfactory': "If you are unusually sensitive to air pollution, consider reducing prolonged outdoor exertion.",
                'Moderate': "If you are unusually sensitive to air pollution, consider reducing prolonged outdoor exertion.",
                'Poor': "Minimize outdoor activities, especially if you have respiratory or heart problems.",
                'Very Poor': "Avoid all outdoor activities, and stay indoors with windows and doors closed. Use air purifiers if available.",
                'Severe': "\n1. Stay indoors as much as possible.\n2. Use air purifiers indoors.\n3. Keep windows and doors closed.\n4. Wear N95 masks if you must go outside.\n5. Stay hydrated and maintain a healthy diet."
            }
            
            for category, (lower, upper) in aqi_ranges.items():
                if lower <= aqi_value <= upper:
                    description = descriptions.get(category, "Description not available.")
                    recommendation = recommendations.get(category, "Recommendation not available.")
                    return category, description, recommendation
        
        if st.button('Predict AQI'):
            input_features_aqi = preprocess_input_aqi(st.session_state.pm25, st.session_state.pm10, st.session_state.no, st.session_state.no2, st.session_state.nox, st.session_state.co, st.session_state.so2)
            predicted_aqi = rf_regressor.predict(input_features_aqi)
            formatted_aqi = f'{predicted_aqi[0]:.2f}'
            aqi_category, aqi_description, aqi_recommendation = classify_aqi(float(formatted_aqi))
            st.success(f'The predicted AQI is: {formatted_aqi} ({aqi_category})\n\n{aqi_description}\n\nRecommendations:\n{aqi_recommendation}')
            if (aqi_category == 'Good'):
                set_background('Good.png')
            elif (aqi_category == 'Satisfactory'):
                set_background('Satisfactory.png')
            elif (aqi_category == 'Moderate'):
                set_background('Moderate.png')
            elif (aqi_category == 'Poor'):
                set_background('Poor.png')
            elif (aqi_category == 'Very Poor'):
                set_background('Very_Poor.png')
            else:
                set_background('Severe.png')


        st.button('Reset AQI Inputs', on_click=reset_aqi_callback)

    st.header("⏰ Predict Best Time of Day for Air Quality")
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            selected_city = st.selectbox('City', city_names, index=city_names.index(st.session_state.city), key='city')
            city_index = city_names.index(selected_city)
        with col2:
            selected_date = st.date_input('Select Date', min_value=pd.Timestamp.now().date(), max_value=pd.to_datetime('2030-12-31'), value=st.session_state.date, key='date')
        
        def get_season(month):
            if month in [12, 1]:
                return 'Winter'
            elif month in [2, 3]:
                return 'Spring'
            elif month in [4, 5]:
                return 'Pre-Monsoon (Summer)'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            elif month in [10, 11]:
                return 'Post-Monsoon (Autumn)'
            else:
                return 'Unknown'

        def get_time_category(predicted_time):
            time_ranges = {
                'Dawn': range(6, 8),
                'Morning': range(8, 12),
                'Afternoon': range(12, 14),
                'Dusk': range(14, 16),
                'Night': list(range(16, 24)) + list(range(0, 6))
            }

            for time_category, time_range in time_ranges.items():
                if predicted_time in time_range:
                    return time_category

            return 'Unknown'

        def get_day_category(predicted_day):
            day_categories = {
                '0': 'Dawn',
                '1': 'Morning',
                '2': 'Afternoon',
                '3': 'Dusk',
                '4': 'Night',
            }
            return day_categories.get(str(predicted_day), 'Unknown')
            
        with col3:
            if st.button('Predict Best Time of Day'):
                selected_month = selected_date.month
                selected_day = selected_date.day
                input_features_time = preprocess_input_time(city_index, selected_month, selected_day)
                predicted_time = dt_classifier.predict(input_features_time)
                best_time_category = get_day_category(predicted_time[0])
                season = get_season(selected_month)
                st.success(f'The Best Time to go out in {selected_city} on {selected_date.strftime("%B %d")} ({season}) is in the {best_time_category}.')
                if (best_time_category == 'Dawn'):
                    set_background('Dawn.png')
                elif (best_time_category == 'Morning'):
                    set_background('Morning.png')
                elif (best_time_category == 'Afternoon'):
                    set_background('Afternoon.png')
                elif (best_time_category == 'Dusk'):
                    set_background('Dusk.png')
                else:
                    set_background('Night.png')
                
            st.button('Reset Time Inputs', on_click=reset_time_callback)


if __name__ == '__main__':
    main()
