🎨 AI Ghibli Style Image Trends – Machine Learning Project
Author: Shoq Eqab Ahmed Alhossainy
Dataset Source: Kaggle – AI Ghibli Style Image Trends 2025
GitHub Repository: ai_ghibli_trend_dataset_v

🎯 Objectives
Explore engagement patterns for AI-generated Ghibli-style images.
Apply multiple ML algorithms for regression and comparison.
Identify the best-performing model for predicting user engagement.
Visualize relationships among features such as generation time, platform, and engagement metrics.

🧠 Dataset Information
Name: AI Ghibli Style Image Trends Dataset
Samples: 10,000 images
Features: 16 (numerical + categorical)
Problem Type: Regression (predicting number of likes)
Source: Kaggle

Main Attributes:
| Feature                 | Description                    |
| ----------------------- | ------------------------------ |
| Platform                | Instagram, TikTok, Twitter     |
| Likes, Shares, Comments | Engagement metrics             |
| Generation Time         | Time taken to create the image |
| File Size, Resolution   | Technical attributes           |
| Creation Date           | Used for time-based analysis   |

⚙️ Data Preprocessing
Missing Values:
Numerical → Filled with median
Categorical → Filled with most frequent value
Encoding: Label Encoding for categorical features
Scaling: StandardScaler for normalization
Feature Engineering: Extracted date components (year, month, day)
Train-Test Split: 70% training / 30% testing (random_state=42)

🤖 Machine Learning Models Applied
| Model             | MSE ↓   | R² ↑                              | Notes                          |
| ----------------- | ------- | --------------------------------- | ------------------------------ |
| Linear Regression | 1200    | 0.65                              | Baseline model                 |
| Random Forest     | 850     | 0.78                              | Strong ensemble method         |
| SVM               | 1100    | 0.68                              | Moderate performance           |
| Gradient Boosting | **800** | **0.80**                          | ✅ Best-performing model        |
| KNN               | -       | Low accuracy                      | Struggled with high dimensions |
| ANN               | -       | Good learning but higher variance |                                |

🏆 Best Model: Gradient Boosting
Handles non-linear relationships effectively.
Reduces overfitting better than Random Forest.

📊 Visualizations
Likes Distribution by Platform → Instagram leads in engagement.
Generation Time vs. Likes → Optimal time: 30–60 seconds.
Correlation Heatmap → Likes correlate strongly with shares & comments.
Feature Importance Chart → Visualizes key contributors using Gradient Boosting.

🧩 Key Insights
Instagram posts received the highest average likes.
Medium generation times (30–60 sec) perform best.
Hand-edited images gain ~15% more engagement.
Ensemble methods outperform linear and distance-based algorithms.

📁 Project Structure
ai_ghibli_trend_dataset_v/
│
├── original_data/              # Raw dataset files
├── preprocessed_data/          # Cleaned & processed data
├── real targets for training/  # True engagement labels
├── Results/                    # Output graphs, model scores, and visualizations
│
├── AI GAIBLI.py                # Initial analysis script
├── ai_ghibli_trend_dataset_v.py # Main Python file (final model + evaluation)
├── README.md                   # Project documentation


🧰 Tools & Libraries
Languages: Python
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Environment: Jupyter Notebook / Google Colab

🚀 How to Run
1. Clone the repository:
git clone https://github.com/enshoq/ai_ghibli_trend_dataset_v.git
cd ai_ghibli_trend_dataset_v
2. Install dependencies:
pip install -r requirements.txt
3. Open the Jupyter notebook:
jupyter notebook
4. Run ai_ghibli_analysis.ipynb to reproduce the analysis and results.

💡 Future Improvements
Integrate deep learning (CNN) for visual feature extraction.
Deploy the model with a web dashboard (e.g., Streamlit).
Analyze additional engagement metrics like saves or reposts.

📜 License
This project is open-source and available for educational and research purposes.

