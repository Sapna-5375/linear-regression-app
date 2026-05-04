import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# --- Modern UI Configuration ---
st.set_page_config(page_title="Linear Regression Masterclass", layout="wide")

# Custom CSS for a professional, modern, and vibrant dashboard look
st.markdown("""
    <style>
    /* Global App Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0b0f19 0%, #1a1b41 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, p {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    h1 { color: #ffffff !important; font-weight: 700; }
    h2, h3 { color: #f8fafc !important; }

    /* Primary Buttons */
    div.stButton > button:first-child { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff !important; 
        font-weight: 600; 
        border: none;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.5);
    }

    /* Informative Explanation Boxes */
    .explanation-box { 
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px; 
        border-radius: 16px; 
        border-left: 6px solid #10b981; 
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); 
        margin-bottom: 24px; 
        color: #f1f5f9;
        font-size: 1.05em;
        line-height: 1.6;
        transition: transform 0.2s ease;
    }
    .explanation-box:hover {
        transform: translateY(-2px);
        border-left: 6px solid #34d399;
    }
    .explanation-box b {
        color: #10b981;
        font-size: 1.1em;
        margin-bottom: 8px;
        display: inline-block;
    }

    /* Metric Cards */
    .metric-card { 
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        padding: 20px; 
        border-radius: 16px; 
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); 
        text-align: center; 
        color: #f8fafc;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
        box-shadow: 0 8px 32px 0 rgba(124, 58, 237, 0.2); 
        transform: translateY(-3px);
    }
    
    /* Upload Box Styling */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(30, 41, 59, 0.4);
        border: 2px dashed rgba(124, 58, 237, 0.4);
        border-radius: 16px;
        padding: 20px;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #7c3aed;
        background-color: rgba(124, 58, 237, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation (Pipeline Flow) ---
st.sidebar.title("🚀 ML Learning Pipeline")
st.sidebar.markdown("Follow the steps sequentially to build your model.")
tabs = ["📂 1. Dataset Input", "🛠 2. Preprocessing", "📊 3. Exploratory Data Analysis", "🧠 4. Learning Module", "⚙️ 5. Training & Evaluation", "🔮 6. Prediction & Inference"]
choice = st.sidebar.radio("Navigation", tabs)

# --- 1. DATASET INPUT MODULE ---
if choice == "📂 1. Dataset Input":
    st.header("Step 1: Dataset Input Module")
    st.markdown('<div class="explanation-box"><b>Why this step?</b> Linear Regression requires data to learn patterns. By uploading your CSV, we can inspect the raw information and choose which "features" (inputs) help us predict the "target" (output).</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Preview (Head)")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.subheader("Dataset Summary")
            st.write(f"**Total Rows:** {df.shape[0]}")
            st.write(f"**Total Columns:** {df.shape[1]}")
            st.write("**Column Data Types:**")
            st.write(df.dtypes)
    else:
        st.info("👋 Welcome! Please upload a CSV file (e.g., Boston Housing) to get started.")

# --- 2. PREPROCESSING INTERFACE ---
elif choice == "🛠 2. Preprocessing":
    st.header("Step 2: Preprocessing Interface")
    if 'df' not in st.session_state:
        st.error("Please upload a dataset in Step 1 first!")
    else:
        df = st.session_state['df'].copy()
        
        st.subheader("Configure Features & Target")
        target = st.selectbox("Select Target Variable (Y) - What to predict?", df.columns)
        features = st.multiselect("Select Feature Variables (X) - What to use for prediction?", [c for c in df.columns if c != target])
        
        if features:
            st.markdown("---")
            st.subheader("Interactive Cleaning Options")
            
            c1, c2 = st.columns(2)
            with c1:
                with st.expander("🛠 Missing Value Handling"):
                    st.write("**What:** Replacing 'NaN' (Not a Number) entries.")
                    st.write("**Why:** Math equations cannot process empty cells. We use 'Mean Imputation' to fill gaps without changing the column average.")
                    if st.button("Apply Mean Imputation"):
                        cols_to_fill = features + [target]
                        df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mean())
                        st.success("Cleaned missing values in features and target!")
            
            with c2:
                with st.expander("⚖️ Feature Scaling"):
                    st.write("**What:** Putting all numbers on a similar scale (e.g., 0 to 1).")
                    st.write("**Why:** It prevents variables with huge numbers (like 'Salary') from overpowering smaller numbers (like 'Age') during calculation.")
                    scale_opt = st.selectbox("Method", ["None", "Standardization", "Normalization"])
                    if scale_opt == "Standardization":
                        df[features] = (df[features] - df[features].mean()) / df[features].std()
                    elif scale_opt == "Normalization":
                        df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
            
            st.session_state['processed_df'] = df
            st.session_state['features'] = features
            st.session_state['target'] = target
            
            st.subheader("Processed Data Comparison")
            st.write("Preview of your data after cleaning and scaling:")
            st.dataframe(df[features + [target]].head())

# --- 3. EXPLORATORY DATA ANALYSIS ---
elif choice == "📊 3. Exploratory Data Analysis":
    st.header("Step 3: Exploratory Data Analysis (EDA)")
    if 'processed_df' not in st.session_state:
        st.error("Please process your data in Step 2 first!")
    else:
        df = st.session_state['processed_df']
        feat, target = st.session_state['features'], st.session_state['target']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Distributions")
            st.write("Histograms show the frequency of values.")
            f_sel = st.selectbox("Select Feature to View", feat)
            fig = px.histogram(df, x=f_sel, marginal="box", title=f"Distribution of {f_sel}")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Correlation Analysis")
            st.write("Heatmaps show how strongly variables relate to each other.")
            numeric_df = df[feat + [target]].select_dtypes(include=[np.number])
            fig_heat = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heat, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Relationship Visualization (Feature vs Target)")
        rel_feat = st.selectbox("Select Feature to check Correlation with Target", feat)
        fig_rel = px.scatter(df, x=rel_feat, y=target, trendline="ols", title=f"Linear Relationship: {rel_feat} vs {target}")
        st.plotly_chart(fig_rel, use_container_width=True)

# --- 4. LEARNING MODULE ---
elif choice == "🧠 4. Learning Module":
    st.header("Step 4: Linear Regression Learning Module")
    
    st.markdown('<div class="explanation-box"><h3>The Math Behind the Model</h3>Linear Regression is the process of finding the "best-fit" line through your data. This module explains the mathematical blueprint step-by-step.</div>', unsafe_allow_html=True)
    
    st.subheader("A. Simple vs Multiple Linear Regression")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><h4>Simple Linear Regression</h4>Uses exactly <b>one</b> feature to predict the target.</div>', unsafe_allow_html=True)
        st.latex(r"y = \theta_0 + \theta_1 x_1")
    with col2:
        st.markdown('<div class="metric-card"><h4>Multiple Linear Regression</h4>Uses <b>two or more</b> features to predict the target.</div>', unsafe_allow_html=True)
        st.latex(r"y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n")

    st.markdown("---")
    
    st.subheader("1. Hypothesis Formulation")
    st.latex(r"h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n")
    st.info("The hypothesis $h_\\theta(x)$ is the model's prediction. It multiplies each input feature ($x_i$) by a learned parameter/weight ($\\theta_i$) and adds a bias ($\\theta_0$).")
    
    st.subheader("2. Cost Function Definition (MSE)")
    st.latex(r"J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2")
    st.info("The cost function $J(\\theta)$ calculates the Mean Squared Error (MSE). It measures how far off the model's predictions are from the actual values ($y$). The goal of learning is to minimize this cost.")

    st.subheader("3. Error Computation & Parameter Learning (Gradient Descent)")
    st.latex(r"\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)")
    st.latex(r"\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}")
    st.info("To learn the best parameters, the model uses Gradient Descent. It calculates the derivative (gradient) of the cost function to find the direction of steepest descent, then updates each $\\theta_j$ by taking a step of size $\\alpha$ (learning rate) in the opposite direction.")

    st.markdown("---")
    
    st.subheader("🔬 Step-by-Step Computation (Live Sample)")
    if 'processed_df' not in st.session_state or 'features' not in st.session_state:
        st.warning("Please setup your data in Steps 1 & 2 to see a live computation sample!")
    else:
        df = st.session_state['processed_df']
        feat = st.session_state['features']
        target = st.session_state['target']
        
        if len(feat) == 0:
            st.error("No features selected.")
        else:
            # Take the first row of data
            sample_row = df.iloc[0]
            
            st.markdown("Let's look at the **first row** of your processed dataset to see how the math works in practice. We will assume some random initial parameters ($\\theta$) for this demonstration:")
            
            dummy_bias = 0.5
            dummy_weights = [round(0.2 * (i+1), 2) for i in range(len(feat))]
            
            st.latex(r"\text{Assume } \theta_0 = " + str(dummy_bias))
            weight_strs = [fr"\theta_{i+1} = {w}" for i, w in enumerate(dummy_weights)]
            st.latex(", \\; ".join(weight_strs))
            
            st.markdown("**1. Formula Used (Hypothesis):**")
            hypothesis_str = f"h_\\theta(x) = {dummy_bias}"
            for i, f in enumerate(feat):
                val = round(sample_row[f], 4)
                hypothesis_str += f" + ({dummy_weights[i]} \\times {val})"
            st.latex(hypothesis_str)
            
            st.markdown("**2. Intermediate Values:**")
            pred = dummy_bias
            calc_steps = []
            for i, f in enumerate(feat):
                val = round(sample_row[f], 4)
                term = dummy_weights[i] * val
                pred += term
                calc_steps.append(f"\\text{{Term }}_{i+1} ({f}): {dummy_weights[i]} \\times {val} = {round(term, 4)}")
            
            for step in calc_steps:
                st.latex(step)
                
            st.latex(f"\\text{{Sum of intermediate values (Prediction) }} h_\\theta(x) = {round(pred, 4)}")
            
            actual_y = round(sample_row[target], 4)
            st.markdown("**3. Error Computation:**")
            st.latex(f"\\text{{Actual Value }} (y) = {actual_y}")
            
            error = pred - actual_y
            sq_error = error ** 2
            st.latex(f"\\text{{Error }} = h_\\theta(x) - y = {round(pred, 4)} - {actual_y} = {round(error, 4)}")
            st.latex(f"\\text{{Squared Error }} = ({round(error, 4)})^2 = {round(sq_error, 4)}")
            
            st.markdown('<div class="explanation-box"><b>Learning Step:</b> The model would compute this squared error across all samples, average it (MSE), and use the gradient to adjust the $\\theta$ parameters to reduce the error on the next pass!</div>', unsafe_allow_html=True)

# --- 5. TRAINING & EVALUATION ---
elif choice == "⚙️ 5. Training & Evaluation":
    st.header("Step 5: Model Training & Evaluation")
    if 'processed_df' not in st.session_state:
        st.error("Please setup your data first!")
    else:
        df = st.session_state['processed_df']
        feat, target = st.session_state['features'], st.session_state['target']
        X, y = df[feat], df[target]
        
        # User Controlled Config
        split = st.slider("Train-Test Split Ratio (Training %)", 0.5, 0.9, 0.8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        # Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.session_state['model'] = model
        
        st.markdown("---")
        st.subheader("1. Final Learned Parameters ($\\theta$)")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write(f"**Intercept (Bias - $\\theta_0$):** `{model.intercept_:.4f}`")
        for i, f in enumerate(feat):
            st.write(f"**Coefficient for {f} (Weight - $\\theta_{i+1}$):** `{model.coef_[i]:.4f}`")
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("2. Model Accuracy Metrics")
        preds = model.predict(X_test)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("R² Score (Goodness of Fit)", round(r2_score(y_test, preds), 4))
        with m2: st.metric("Mean Squared Error (MSE)", round(mean_squared_error(y_test, preds), 4))
        with m3: st.metric("Mean Absolute Error (MAE)", round(mean_absolute_error(y_test, preds), 4))

        st.markdown("---")
        st.subheader("3. Model Visualization")
        
        if len(feat) == 1:
            st.write(f"**Single Feature Regression Line:** `{feat[0]}` vs `{target}`")
            fig = px.scatter(x=X_test[feat[0]], y=y_test, title="Test Data vs Model Prediction")
            fig.add_trace(go.Scatter(x=X_test[feat[0]], y=preds, mode='lines', name='Regression Line', line=dict(color='#7c3aed', width=3)))
            fig.update_layout(xaxis_title=feat[0], yaxis_title=target)
            st.plotly_chart(fig, use_container_width=True)
            
        elif len(feat) == 2:
            st.write(f"**Multi-dimensional Visualization (3D Plane):** `{feat[0]}` & `{feat[1]}` vs `{target}`")
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=X_test[feat[0]], y=X_test[feat[1]], z=y_test, mode='markers', name='Actual Data', marker=dict(size=4, color='#10b981')))
            
            # Create a meshgrid for the plane
            x_min, x_max = X_test[feat[0]].min(), X_test[feat[0]].max()
            y_min, y_max = X_test[feat[1]].min(), X_test[feat[1]].max()
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            
            # Predict z values for the mesh
            z_grid = model.intercept_ + model.coef_[0] * x_grid + model.coef_[1] * y_grid
            
            fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, name='Regression Plane', opacity=0.6, colorscale='Purples'))
            fig.update_layout(scene=dict(xaxis_title=feat[0], yaxis_title=feat[1], zaxis_title=target), title="3D Scatter with Hyperplane")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Visualizing {len(feat)} dimensions is not natively supported in 3D plots. We recommend keeping it to 1 or 2 features for visual interpretation.")
            
        st.markdown("---")
        st.subheader("4. Cost Convergence (Gradient Descent Simulation)")
        st.markdown('<div class="explanation-box">LinearRegression uses exact Ordinary Least Squares (OLS) which calculates parameters instantly without iterations. To visualize how a model <b>learns over time</b>, we can simulate Gradient Descent below:</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: learning_rate = st.slider("Learning Rate (alpha)", 0.001, 0.5, 0.1, format="%.3f")
        with c2: epochs = st.slider("Epochs (Iterations)", 10, 500, 100)
        
        if st.button("Simulate Gradient Descent & Plot Convergence"):
            # Simple GD Simulation
            X_gd = X_train.values
            y_gd = y_train.values.reshape(-1, 1)
            m = len(y_gd)
            
            # Add bias column
            X_b = np.c_[np.ones((m, 1)), X_gd]
            theta = np.zeros((X_b.shape[1], 1))
            
            cost_history = []
            
            # Standardize X for GD stability if not done
            X_b_scaled = X_b.copy()
            if X_b.shape[1] > 1:
                X_b_scaled[:, 1:] = (X_b[:, 1:] - np.mean(X_b[:, 1:], axis=0)) / (np.std(X_b[:, 1:], axis=0) + 1e-8)
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(epochs):
                predictions = X_b_scaled.dot(theta)
                errors = predictions - y_gd
                cost = (1 / (2 * m)) * np.sum(errors ** 2)
                cost_history.append(cost)
                
                gradients = (1 / m) * X_b_scaled.T.dot(errors)
                theta = theta - learning_rate * gradients
                
                if epoch % max(1, (epochs // 10)) == 0:
                    progress_bar.progress(min(1.0, (epoch + 1) / epochs))
                    status_text.text(f"Epoch {epoch+1}/{epochs} | Cost: {cost:.4f}")
                    
            progress_bar.progress(1.0)
            status_text.text(f"Training Complete! Final Cost: {cost_history[-1]:.4f}")
            
            fig_cost = px.line(x=range(1, epochs + 1), y=cost_history, title="Cost Function Convergence", labels={'x': 'Epochs (Iterations)', 'y': 'Mean Squared Error (Cost)'})
            fig_cost.update_traces(line=dict(color='#10b981', width=3))
            st.plotly_chart(fig_cost, use_container_width=True)

# --- 6. PREDICTION & INFERENCE ---
elif choice == "🔮 6. Prediction & Inference":
    st.header("Step 6: Prediction & Inference Module")
    if 'model' not in st.session_state:
        st.error("Please train your model in Step 5 first!")
    else:
        model, feat = st.session_state['model'], st.session_state['features']
        
        st.subheader("Manual Inference")
        st.write("Input new values to see how the model calculates a prediction in real-time.")
        
        user_inputs = []
        for f in feat:
            user_inputs.append(st.number_input(f"Enter value for {f}", value=0.0))
            
        if st.button("Calculate Prediction"):
            # The Glass-Box Math Trace
            st.markdown('<div class="explanation-box"><h4>Step-by-Step Prediction Trace</h4>', unsafe_allow_html=True)
            math_str = f"{model.intercept_:.4f}"
            for i, val in enumerate(user_inputs):
                math_str += f" + ({model.coef_[i]:.4f} * {val})"
            
            st.write("**The Formula applied:** Intercept + sum(Weight * Input)")
            st.code(f"Result = {math_str}")
            
            final_pred = model.predict([user_inputs])[0]
            st.write(f"<h3>Final Predicted Value: {final_pred:.4f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)