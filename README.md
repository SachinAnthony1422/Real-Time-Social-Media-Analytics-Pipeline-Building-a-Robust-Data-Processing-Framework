## 🏡 Real-Time Social Media Analytics Pipeline

### 📌 Overview
This project aims to build a **real-time social media analytics pipeline** to extract meaningful insights from various platforms. It includes **sentiment analysis, hashtag clustering, and engagement prediction** using machine learning models.

The system processes high-velocity data, enabling businesses and researchers to monitor trends and user engagement efficiently.

### 🚀 Tech Stack Used
🐍 **Python** – for data processing and model training  
📊 **Google Colab** – for dataset analysis and model development  
🌐 **Flask** – for web framework and deployment  
🤖 **Scikit-Learn, TensorFlow** – for machine learning models  
📈 **Pandas, NumPy** – for data manipulation  
📉 **Matplotlib, Seaborn, Plotly** – for data visualization  
💾 **Joblib** – for model saving/loading  

### 📂 Dataset
The dataset includes various features such as:
✔ Social media text content  
✔ Hashtags and keywords  
✔ Likes, shares, and comments  
✔ User engagement metrics  
✔ Sentiment scores  

### 🛠 Preprocessing Steps:
✅ Handling missing values  
✅ Encoding categorical variables  
✅ Normalizing numerical features  
✅ Removing stopwords and special characters from text  

### 🔥 Model Training Process
#### 📌 Data Preprocessing:
✅ Tokenization and text cleaning  
✅ Feature scaling for numerical data  

#### 🏆 Model Selection & Training:
✅ **Sentiment Analysis** using VADER  
✅ **Hashtag Clustering** using TF-IDF and K-Means  
✅ **Engagement Prediction** using Random Forest  
✅ Hyperparameter tuning with GridSearchCV  

#### 📊 Model Evaluation:
✅ **Silhouette Score** for clustering  
✅ **Mean Squared Error (MSE)** for regression models  
✅ **Mean Absolute Error (MAE)**  
✅ **F1 Score, Accuracy** for classification tasks  

#### 📈 Visualization Techniques:
✅ Word Clouds for trending topics  
✅ Correlation heatmaps  
✅ Cluster distribution plots  
✅ Model performance graphs  

### 🌍 Model Deployment using Flask/Django
1️⃣ User inputs text/hashtags via a web form  
2️⃣ Pre-trained models analyze data in real-time  
3️⃣ Results are displayed on the dashboard  

### 🛠 Installation and Setup
```bash
# 1️⃣ Clone the Repository
git clone https://github.com/SachinAnthony1422/Real-Time-Social-Media-Analytics-Pipeline-Building-a-Robust-Data-Processing-Framework.git
cd Real-Time-Social-Media-Analytics-Pipeline-Building-a-Robust-Data-Processing-Framework

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# 3️⃣ Run Flask Server
python app.py
```
🔗 Open `http://127.0.0.1:8000/` in your browser to access the web application.
```bash
git clone https://github.com/SachinAnthony1422/Real-Time-Social-Media-Analytics-Pipeline-Building-a-Robust-Data-Processing-Framework.git
cd Real-Time-Social-Media-Analytics-Pipeline-Building-a-Robust-Data-Processing-Framework
```
#### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
#### 3️⃣ Run Flask Server
```bash
python app.py
```
🔗 Open `http://127.0.0.1:8000/` in your browser to access the web application.

### 💡 Usage
1️⃣ Enter text or hashtag data on the analytics page.  
2️⃣ Click "Analyze" to get insights.  
3️⃣ View sentiment scores, trending hashtags, and engagement predictions.  

### 🔮 Future Enhancements
☁ Deploy model on cloud platforms (AWS, Heroku, or GCP)  
🧠 Implement deep learning for improved predictions  
📊 Integrate real-time API data streaming  
🎨 Enhance UI/UX for a better user experience  

🚀 **Happy Coding!**

