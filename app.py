import streamlit as st
import torch
import numpy as np

# ... (Your existing code for model loading and prediction) ...
# Define your model architecture to match the saved model's architecture
class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 128):  # Added hidden_size argument
        super(MyModel, self).__init__()
        self.input_size = 4
        self.output_size = 6
        self.network = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.output_size),
            nn.Sigmoid(),  # Ensure outputs are between 0 and 1
        )

    def forward(self, x):
        # Define forward pass
        return self.network(x)  # Use the defined network for forward pass


# Load the model and weights
model = MyModel()
model.load_state_dict(torch.load("wormhole_stability_model.pth"))
model.eval()


# Define prediction function
@st.cache_resource  # Cache the model loading to improve performance
def predict(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.numpy()


# Initialize session state for the counter
if 'counter' not in st.session_state:
    st.session_state.counter = 0


# Function to increment the counter
def increment_counter():
    st.session_state.counter += 1


st.title("Wormhole Stability Prediction App")

# Display the counter value
st.write(f"Counter: {st.session_state.counter}")

# Button to increment the counter
if st.button("Increment"):
    increment_counter()

# Input fields for wormhole parameters
throat_radius = st.number_input("Throat Radius", value=0.5, min_value=0.0, max_value=1.0)
energy_density = st.number_input("Energy Density", value=0.5, min_value=0.0, max_value=1.0)
field_strength = st.number_input("Field Strength", value=0.5, min_value=0.0, max_value=1.0)
temporal_flow = st.number_input("Temporal Flow", value=0.5, min_value=0.0, max_value=1.0)

# Make prediction when button is clicked
if st.button("Predict"):
    input_data = [throat_radius, energy_density, field_strength, temporal_flow]
    prediction = predict(input_data)

    # Display prediction results
    st.subheader("Prediction Results:")
    st.write(f"Lambda: {prediction[0][0]:.4f}")
    st.write(f"Stability: {prediction[0][1]:.4f}")
    st.write(f"Coherence: {prediction[0][2]:.4f}")
    st.write(f"Wormhole Integrity: {prediction[0][3]:.4f}")
    st.write(f"Field Alignment: {prediction[0][4]:.4f}")
    st.write(f"Temporal Coupling: {prediction[0][5]:.4f}")
