import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------
# 🔧 Train and Save TF-IDF + KNN job recommendation model
# -------------------------------------------------------

def train_recommender(data_path="data/job_dataset_100.csv"):
    df = pd.read_csv(data_path)

    # ✅ Remove extra spaces from column names
    df.columns = [col.strip() for col in df.columns]

    # ✅ Fill missing text fields
    df['Skills'] = df['Skills'].fillna('')
    df['Job Title'] = df['Job Title'].fillna('')

    # ✅ Combine Skills + Job Title for TF-IDF input
    df['combined'] = df['Skills'] + " " + df['Job Title']
    df = df[df['combined'].str.strip() != '']  # remove empty rows

    # ✅ Train TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['combined'])

    # ✅ Use min(5, number of rows) to avoid errors
    num_neighbors = min(5, X.shape[0])
    model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    model.fit(X)

    # ✅ Save models
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_model.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("models/knn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return "✅ Model trained and saved successfully."


# ------------------------------------------------------------
# 📌 Recommend jobs based on user input using saved models
# ------------------------------------------------------------
def recommend_jobs(user_input, data_path="data/job_dataset_100.csv"):
    # ✅ Load models
    with open("models/tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/knn_model.pkl", "rb") as f:
        model = pickle.load(f)

    # ✅ Load dataset
    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]
    df['Skills'] = df['Skills'].fillna('')
    df['Job Title'] = df['Job Title'].fillna('')
    df['combined'] = df['Skills'] + " " + df['Job Title']
    df = df[df['combined'].str.strip() != '']

    # ✅ Transform input & find neighbors
    user_vector = tfidf.transform([user_input])
    distances, indices = model.kneighbors(user_vector)

    # ✅ Return more useful info (Category, Demand Level, Expected Salary)
    results = df.iloc[indices[0]][[
        'Job Title', 'Skills', 'Category', 'Demand Level', 'Expected Salary', 'Updated At'
    ]]
    return results.reset_index(drop=True)
















































# import os
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors

# # -------------------------------------------------------
# # 🔧 Train and Save TF-IDF + KNN job recommendation model
# # -------------------------------------------------------

# def train_recommender(data_path = "data/job_dataset_100.csv"
# ):
#     import os
#     df = pd.read_csv(data_path)

#     df['Skills'] = df['Skills'].fillna('')
#     df['Job Title'] = df['Job Title'].fillna('')
#     df['combined'] = df['Skills'] + " " + df['Job Title']
#     df = df[df['combined'].str.strip() != '']

#     tfidf = TfidfVectorizer()
#     X = tfidf.fit_transform(df['combined'])

#     # ✅ Limit neighbors to number of available samples
#     num_samples = X.shape[0]
#     num_neighbors = min(5, num_samples)  # Use 5 or fewer if dataset is small

#     model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
#     model.fit(X)

#     os.makedirs("models", exist_ok=True)

#     with open("models/tfidf_model.pkl", "wb") as f:
#         pickle.dump(tfidf, f)
#     with open("models/knn_model.pkl", "wb") as f:
#         pickle.dump(model, f)

#     return "✅ Model trained and saved successfully."


# # ------------------------------------------------------------
# # 📌 Recommend jobs based on user input using saved models
# # ------------------------------------------------------------
# def recommend_jobs(user_input, data_path = "data/job_dataset_100.csv"):
#     # Load saved TF-IDF vectorizer and KNN model
#     with open("models/tfidf_model.pkl", "rb") as f:
#         tfidf = pickle.load(f)
#     with open("models/knn_model.pkl", "rb") as f:
#         model = pickle.load(f)

#     # Load and clean the dataset again
#     df = pd.read_csv(data_path)
#     df['Skills'] = df['Skills'].fillna('')
#     df['Job Title'] = df['Job Title'].fillna('')
#     df['combined'] = df['Skills'] + " " + df['Job Title']
#     df = df[df['combined'].str.strip() != '']

#     # Convert user input into the same TF-IDF space
#     user_vector = tfidf.transform([user_input])

#     # Find top 5 similar job roles
#     distances, indices = model.kneighbors(user_vector)

#     # Return recommended jobs (Job Title, Skills, Category)
#     results = df.iloc[indices[0]][['Job Title', 'Skills', 'Category']]
#     return results.reset_index(drop=True)














































