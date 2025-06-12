import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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

# Sidebar Branding with Placeholder Logo
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3945/3945685.png", width=100)
st.sidebar.title("🏡 About This App")
st.sidebar.info("""
This app predicts the price of a residential property based on features like location, size, type, etc.

Developed by: **Mr. Singh**  
Data Source: Internal Real Estate Dataset  
Model: Trained ML Pipeline  
""")

# Load pickled files
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Main Title
st.title("🏢 Real Estate Price Predictor")
st.subheader("Fill in the details below to get a predicted price range")

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
        servant_room_input = st.selectbox('Servant Room', ['No', 'Yes'])
        store_room_input = st.selectbox('Store Room', ['No', 'Yes'])

        # Convert to float
        servant_room = 1.0 if servant_room_input == 'Yes' else 0.0
        store_room = 1.0 if store_room_input == 'Yes' else 0.0

        furnishing_type = st.selectbox('🛋️ Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
        luxury_category = st.selectbox('💎 Luxury Category', sorted(df['luxury_category'].unique().tolist()))
        floor_category = st.selectbox('🏢 Floor Category', sorted(df['floor_category'].unique().tolist()))

    submitted = st.form_submit_button("🔮 Predict Price")

# Prediction logic
if submitted:
    # Create input DataFrame
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age,
             built_up_area, servant_room, store_room,
             furnishing_type, luxury_category, floor_category]]

    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = round(base_price - 0.11, 2)
    high = round(base_price + 0.11, 2)

    # Display result
    st.markdown("---")
    st.success("🏠 Prediction Completed!")
    st.markdown("### 💰 Estimated Price Range (in Cr):")
    st.metric(label="Lower Bound", value=f"{low} Cr")
    st.metric(label="Upper Bound", value=f"{high} Cr")

    # SHAP Explainability
    st.markdown("---")
    st.markdown("### 📊 Feature Importance (SHAP)")

    try:
        # Extract model and preprocessor
        model = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']

        # Transform the input
        transformed_data = preprocessor.transform(one_df)

        # Get actual feature names after preprocessing
        def get_feature_names(preprocessor, input_df):
            output_features = []

            for name, transformer, cols in preprocessor.transformers_:
                if name != 'remainder':
                    if hasattr(transformer, 'get_feature_names_out'):
                        names = transformer.get_feature_names_out(cols)
                    else:
                        names = cols
                    output_features.extend(names)
                else:
                    output_features.extend(cols)
            return output_features

        feature_names = get_feature_names(preprocessor, one_df)

        # SHAP Explainer
        explainer = shap.Explainer(model, feature_names=feature_names)
        shap_values = explainer(transformed_data)

        # Plot SHAP bar chart
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation not available: " + str(e))
