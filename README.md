# Career Path & Job Recommender 🎯💼

This project is a **Career Path & Job Recommendation System** built with **Streamlit** and **Machine Learning**.
It suggests relevant job roles/career paths based on a user’s **skills, interests, or experience**.

---

## 📂 Project Structure

```
Career path and job recommender/
│── data/
│   └── job_dataset_100.csv        # Sample dataset of jobs & skills
│
│── models/
│   ├── knn_model.pkl              # Trained KNN model
│   └── tfidf_model.pkl            # TF-IDF vectorizer model
│
│── utils/
│   └── preprocessing.py           # Text preprocessing (cleaning user input)
│
│── app.py                         # Streamlit frontend app
│── recommender.py                 # Training & job recommendation logic
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository** (or download the folder):

   ```bash
   git clone https://github.com/yourusername/career-job-recommender.git
   cd career-job-recommender
   ```

2. **Create virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate      # For Linux/Mac
   venv\Scripts\activate         # For Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the link shown in the terminal (e.g., `http://localhost:8501`) in your browser.

---

## 🛠 How It Works

1. **Preprocessing**

   * User input (skills/interests) is cleaned using regex (removes special chars, converts to lowercase).

2. **TF-IDF Vectorization**

   * Converts text into numerical vectors representing importance of words.

3. **K-Nearest Neighbors (KNN) Algorithm**

   * Finds the most similar job roles in the dataset based on TF-IDF vectors.

4. **Recommendation Output**

   * Displays a dataframe with suggested jobs or career paths.

---

## 📖 Example Usage

* Input:

  ```
  "I love coding, AI, and problem solving"
  ```

* Output:
  ✅ Software Engineer
  ✅ AI Researcher
  ✅ Data Scientist

---

## 📌 Files Overview

* **app.py** → Streamlit UI (user input, button, and results).
* **recommender.py** → Core ML logic (training + job recommendation).
* **preprocessing.py** → Cleans and preprocesses text input.
* **job\_dataset\_100.csv** → Sample dataset of jobs and related skills.

---

## 🔮 Future Improvements

* Add **more datasets** (e.g., LinkedIn or Kaggle job datasets).
* Implement **Deep Learning (Transformers)** for better recommendations.
* Include **job descriptions** in results.
* Deploy online (Streamlit Cloud / Heroku / AWS).
