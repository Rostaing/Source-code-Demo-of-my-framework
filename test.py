# WEB MODE - Streamlit App

import streamlit as st
from rostaingchain import RostaingAgent
from dotenv import load_dotenv

import pandas as pd
import os

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Rostaing AI", page_icon="🤖")

load_dotenv()
llm_api_key = os.getenv("GROQ_API_KEY")

# ----------------------
# CACHE DATA + AGENT
# ----------------------
@st.cache_resource
def load_agent():

    # df = pd.read_csv("C:/Users/Rostaing/Desktop/db/weather_classification_data.csv")

    # doc = "C:/Users/Rostaing/Desktop/db/img1.webp"
    # tub = "https://www.youtube.com/watch?v=reuDYC2zdns"

    # Configuration SQL
    # config_postgres = {
    #     "type": "sql",
    #     # Format: postgresql+psycopg2://user:password@host:port/dbname
    #     "connection_string": "postgresql+psycopg2://postgres:rostaing@localhost:5432/popu_db",
    #     "query": "SELECT * FROM salaire"
    # }

    agent = RostaingAgent(
        llm_model="openai/gpt-oss-120b",  # "openai/gpt-oss-120b",
        llm_provider="groq",
        llm_api_key=llm_api_key,
        data_source="app.py",  # Can be a path to CSV/SQL/Excel/Image/Audio/Video/... or a URL
        vector_db="chroma",                 # High-performance vector storage
        auto_update=True,
        memory=True,
        # reset_db=True,
        save_logs=True,
        save_graph=True
        # security_filters=["MONEY", "SALARY"],
        # output_format="json"
    )

    return agent


agent = load_agent()

# ----------------------
# SESSION STATE
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 RostaingChain Data Assistant")

# ----------------------
# DISPLAY CHAT HISTORY
# ----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------
# USER INPUT
# ----------------------
prompt = st.chat_input("Pose une question sur les données...")

if prompt:
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse assistant (sans streaming)
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            response = agent.chat(prompt)

        st.markdown(response)

    # Sauvegarde réponse
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# ------------------------------------------------------------------- #

# LOCAL MODE - Console App

# import os
# from dotenv import load_dotenv
# from rostaingchain import RostaingAgent
# import os

# import pandas as pd
# import polars as pl

# # 1. Load environment variables (Make sure your .env file is set up)
# load_dotenv()


# def main():
#     # 2. Initialize the Agent
#     # RostaingAgent automatically handles data profiling, vector indexing, and memory.
#     # df = pd.read_excel("C:/Users/Rostaing/Desktop/db/ds_salaries.csv")

#     agent = RostaingAgent(
#         llm_model="gpt-4o",
#         llm_provider="openai",
#         llm_api_key=os.getenv("OPENAI_API_KEY"),
#         data_source="C:/Users/Rostaing/Desktop/db/titanic.csv",  # Path to your CSV/SQL/Excel/Image/Audio/Video/...
#         vector_db="chroma",                 # High-performance vector storage
#         reset_db=False,                    # Set to True to re-index the data
#         memory=True,                       # Keep track of the conversation context
#         auto_update=True,
#         poll_interval=60,
#         # save_graph=True,
#         save_logs=True
#     )

#     print("\n" + "="*40)
#     print("🤖 RotaingChain: CONSOLE MODE")
#     print("Type your question below or 'q' to exit.")
#     print("="*40 + "\n")

#     try:
#         while True:
#             # 3. Capture User Input
#             user_input = input("👤 You: ").strip()

#             # Exit condition
#             if user_input.lower() in ["q", "quit", "exit"]:
#                 print("\nShutting down... Goodbye! 👋")
#                 break

#             if not user_input:
#                 continue

#             # 4. Generate & Display Response
#             print("🤖 Agent:", end=" ", flush=True)
            
#             try:
#                 response = agent.chat(user_input)
#                 print(response)
#             except Exception as e:
#                 print(f"\n❌ Error: {str(e)}")

#     except KeyboardInterrupt:
#         print("\n\n[System] Session interrupted by user. Closing... 👋")
#     finally:
#         print("Program closed.")

# if __name__ == "__main__":
#     main()


# --------------------------------------------------------------------------- #