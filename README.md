# 📈 Linear Regression Masterclass Dashboard

An interactive, visually stunning Streamlit application designed to teach the fundamentals of Linear Regression. From raw data processing to the mathematical underpinnings of Gradient Descent, this tool offers a complete "glass-box" experience for students and data enthusiasts.

## ✨ Features

- **📂 Dataset Input**: Upload your own CSV datasets, instantly preview data, and analyze schema metadata.
- **🛠 Interactive Preprocessing**: Dynamically handle missing values (Mean Imputation) and scale features (Standardization/Normalization) with interactive toggles.
- **📊 Exploratory Data Analysis (EDA)**: Auto-generate correlation heatmaps and feature vs. target scatter plots to visualize linear relationships using Plotly.
- **🧠 Educational Learning Module**: Dive into the math behind the model. Features side-by-side Simple vs. Multiple Linear Regression comparisons, cost function (MSE) definitions, and a **live step-by-step mathematical trace** using a sample from your actual uploaded data.
- **⚙️ Training & Evaluation**: Configure Train-Test split ratios, fit the model, and review learned parameters ($\theta$) alongside performance metrics (R², MSE, MAE).
- **🔮 Prediction & Inference**: Input custom values to see how the model calculates predictions in real-time.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. 

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Linear_Regression
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 🎨 UI/UX Design

The dashboard features a **Modern Glassmorphism Theme**:
- Deep gradient backgrounds (`#0b0f19` to `#1a1b41`)
- Translucent, frosted glass explanation cards.
- Vibrant, interactive UI components with smooth hover animations.
- Carefully crafted typography using the `Inter` font family.
