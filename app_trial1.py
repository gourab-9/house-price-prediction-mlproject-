import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠", layout="centered")

# Inject custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #1f77b4;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stSelectbox, .stNumberInput {
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏢 Real Estate Price Predictor")
st.subheader("Fill in the details below to get a predicted price range")

# Load pickled files
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        property_type = st.selectbox('🏘️ Property Type', ['flat', 'house'])
        sector = st.selectbox('📍 Sector', sorted(df['sector'].unique().tolist()))
        bedrooms = float(st.selectbox('🛏️ Bedrooms', sorted(df['bedRoom'].unique().tolist())))
        bathroom = float(st.selectbox('🛁 Bathrooms', sorted(df['bathroom'].unique().tolist())))
        balcony = st.selectbox('🏞️ Balconies', sorted(df['balcony'].unique().tolist()))
        property_age = st.selectbox('📅 Property Age', sorted(df['agePossession'].unique().tolist()))

    with col2:
        built_up_area = float(st.number_input('📐 Built Up Area (sq ft)', min_value=0.0))
        servant_room = float(st.selectbox('👨‍🍳 Servant Room', [0.0, 1.0]))
        store_room = float(st.selectbox('📦 Store Room', [0.0, 1.0]))
        furnishing_type = st.selectbox('🛋️ Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
        luxury_category = st.selectbox('💎 Luxury Category', sorted(df['luxury_category'].unique().tolist()))
        floor_category = st.selectbox('🏢 Floor Category', sorted(df['floor_category'].unique().tolist()))

    submitted = st.form_submit_button("🔮 Predict Price")

if submitted:
    # Create input DataFrame
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room,
             store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = round(base_price - 0.11, 2)
    high = round(base_price + 0.11, 2)

    # Result display
    st.markdown("---")
    st.success("🏠 Prediction Completed!")
    st.markdown("### 💰 Estimated Price Range (in Cr):")
    st.metric(label="Lower Bound", value=f"{low} Cr")
    st.metric(label="Upper Bound", value=f"{high} Cr")
