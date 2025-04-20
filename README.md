
🤖 YourBot: Article Research Tool

Welcome to YourBot, a powerful AI-powered equity research assistant. This Streamlit-based web application lets you upload up to 3 news article URLs and query the content using natural language. The tool leverages Google Gemini 1.5 Flash and LangChain to provide intelligent answers with reliable sources—making it perfect for quick and effective article research.

---

🔍 Features

- Input up to 3 article URLs
- Automatically extracts and splits article content into manageable chunks
- Embeds and indexes article text using GoogleGenerativeAI Embeddings
- Asks intelligent questions about uploaded articles using Gemini 1.5 Flash
- Cites the sources used to answer each query
- Clean, responsive Streamlit interface

---

🛠️ Tech Stack

- Streamlit — UI framework
- LangChain — LLM orchestration
- Google Generative AI (Gemini) — LLM and Embeddings
- FAISS — Vector storage and similarity search
- Unstructured — Web article loader
- Tenacity — Robust retry logic
- .env support with python-dotenv

---

⚙️ Setup Instructions

1. Clone the Repository

    git clone https://github.com/Aniru18/Equity_Assistant_YourBot.git

2. Create a Virtual Environment (optional but recommended)

    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. Install Dependencies

    pip install -r requirements.txt

    Note: Make sure you have faiss-cpu, streamlit, langchain, google-generativeai, unstructured, and other required libraries installed.

4. Setup Environment Variables

    Create a .env file in the root directory with your API key:

    GOOGLE_API_KEY=your_google_generative_ai_api_key
## Environment setup:

     conda create -n env_langchain1 python=3.10  
     conda activate env_langchain1
     python -m pip install --upgrade pip
     Install packages:
     pip install -r requirements.txt


---

🚀 Run the App

    streamlit run app3.py

---

🖥️ How to Use

1. Launch the app using the command above.
2. Paste up to 3 article URLs in the sidebar input fields.
3. Click "🚀 Process URLs".
4. Once processed, ask a question related to the content.
5. The app will respond with an answer and source citations.

---

✅ Example Questions

- What is the main theme of the article on Bloomberg?
- What did the CEO say about future plans?
- Summarize the key points from all articles.

---

📜 License

MIT License © Aniruddha Shit

---

🙌 Acknowledgements

- Google Generative AI
- LangChain
- Streamlit
- Unstructured
