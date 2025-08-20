import streamlit as st
import pandas as pd
from recommender import recommend_jobs, train_recommender
from utils.preprocessing import clean_text

st.set_page_config(page_title="Career Recommender", layout="wide")

st.title("🎯 Career Path & Job Recommender")
st.markdown("Enter your interests, skills, or experience to get personalized job suggestions.")

# Training only once
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = train_recommender()

user_input = st.text_area("🧠 Describe your skills/interests")

if st.button("🔍 Recommend Careers"):
    if user_input.strip():
        clean_input = clean_text(user_input)
        results = recommend_jobs(clean_input)
        st.success("Here are some recommended career paths:")
        st.dataframe(results)
    else:
        st.warning("Please enter something about your skills or interests.")
