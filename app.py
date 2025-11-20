import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# ---------------------------
# Load Trained KMeans Model
# ---------------------------
model = pickle.load(open("kmeans_model.pkl", "rb"))

st.title("Customer Segmentation App (KMeans)")
st.write("Enter customer spending values to predict the customer segment.")

# ---------------------------
# Input Fields (6 Features)
# ---------------------------
fresh = st.number_input("Fresh", min_value=0)
milk = st.number_input("Milk", min_value=0)
grocery = st.number_input("Grocery", min_value=0)
frozen = st.number_input("Frozen", min_value=0)
detergents = st.number_input("Detergents_Paper", min_value=0)
delicassen = st.number_input("Delicassen", min_value=0)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Cluster"):

    # User input → array
    user_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])

    # Apply same normalization used during training
    scaled_input = normalize(user_data)

    # Predict the cluster
    cluster_label = model.predict(scaled_input)[0]

    # Meaningful cluster names
    segment_names = {
        0: "Fresh & Frozen Bulk Buyers (Restaurants/Catering)",
        1: "Retail Grocery & Milk Buyers (Supermarkets)"
    }

    st.success(f"Predicted Segment: {segment_names[cluster_label]}")
