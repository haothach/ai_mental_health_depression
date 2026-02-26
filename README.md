# AI Mental Health — Student Depression (Model + Pinecone RAG + Streamlit)

This repository must be run in the following order:

1. **Train/build the ML model** (Notebook: `model.ipynb`)
2. **Build the RAG index in Pinecone**
3. **Run the Streamlit app**

---

## 1) Setup

### Prerequisites
- Windows
- Python 3.10+ (recommended)
- A Pinecone account + API key
- An LLM API key (for example OpenAI)

### Install dependencies
From the project root:

```
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file in the project root (or set variables in your shell). Typical variables:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`

- `OPENAI_API_KEY` (or the key required by your configured LLM provider)

---

## 2) Step 1 — Build the Prediction Model (REQUIRED)

Open and run the notebook:

- **`model.ipynb`**

Run all cells to:
- train the model
- export/save the trained artifacts (model, scaler/encoder, etc.)

After this step, the project should have the saved model artifacts available for the app to load at runtime.

> If the Streamlit app fails with “model not found”, you have not completed this step or the notebook did not export the artifacts to the expected location.

---

## 3) Step 2 — Build Pinecone RAG (REQUIRED)

Prepare your knowledge base documents (your project’s RAG data folder).

Then run the indexing script that:
- chunks documents
- creates/updates the Pinecone index
- uploads embeddings
- (optionally) saves local retrieval artifacts (e.g., BM25 encoder) if your pipeline uses hybrid retrieval

Run (choose the command that matches your project entrypoint):

```
cd src\rag
python build_index.py
```

Alternative (recommended from project root):

```
python -m src.rag.build_index
```

### Expected outputs
After indexing completes, you should have:
- A Pinecone index (e.g. `PINECONE_INDEX_NAME`) containing your uploaded vectors
- Any local hybrid-retrieval artifacts saved (if enabled), such as a BM25 encoder file

> If indexing fails, double-check your `.env` values and confirm your Pinecone project supports the selected `cloud` + `region`.

---

## 4) Step 3 — Run Streamlit (RUN LAST)

Once **Step 1** (model artifacts) and **Step 2** (Pinecone index) are complete, start the Streamlit UI.

From the project root:

```
streamlit run src/streamlit_demo.py
```

If your entrypoint file is different, run that file instead, for example:

```
streamlit run app.py
```

### Using the app
- Open the local URL printed by Streamlit (usually `http://localhost:8501`)
- Follow the chat flow to provide student/profile info
- The app will:
  - run the prediction model
  - optionally retrieve supporting information from Pinecone (RAG)
  - generate an answer using the configured LLM provider

---

## 5) Quick Troubleshooting

### “Model not found” / prediction fails
- Re-run `model.ipynb` and ensure it **saves** the model artifacts.
- Confirm the output artifacts are in the expected folder used by the app.

### Pinecone errors (auth / index / empty results)
- Verify `PINECONE_API_KEY` is correct
- Confirm `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`
- Re-run the indexing step after adding/updating documents

### LLM errors
- Ensure `OPENAI_API_KEY` (or your provider key) is set
- Restart your terminal / VS Code after editing `.env`

---

## 6) Project Structure (high level)

- `model.ipynb` — trains and exports the prediction model
- `src/rag/build_index.py` — builds/updates Pinecone RAG index
- `src/streamlit_demo.py` — Streamlit application entrypoint
- `data/` — datasets + RAG documents
- `model/` — saved artifacts (prediction model, retrieval artifacts, etc.)
