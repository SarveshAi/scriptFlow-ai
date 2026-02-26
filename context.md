# Project Context for ScriptFlow-AI

This file collects the current state and structure of the project to provide a quick overview for future reference.

## Workspace Structure
```
docker-compose.yml
server/
    Dockerfile
    main.py
    requirements.txt
    __pycache__/
    story_db/
        chroma.sqlite3
        97168500-dd6e-434e-88cc-7d7266c4ca56/
```

## Key Files
- `server/main.py`: FastAPI application defining endpoints for adding lore and generating scripts with contextual data. Uses Google Gemini LLM & Chroma vector store.
- `server/requirements.txt`: Lists dependencies including `fastapi`, `uvicorn`, `langchain-*` packages.
- `docker-compose.yml`: Defines container setup (not inspected yet).

## Virtual Environment
- Located at `server/venv`.
- Dependencies installed via `pip install -r requirements.txt`.
- Error encountered due to missing `langchain_google_genai` package in the venv.

## Operational Notes
- Use the venv when running or installing packages.
- Current startup command: `uvicorn main:app --reload`.
- LLM configuration in `main.py` uses environment variable `GOOGLE_API_KEY`.

## Next Steps
- Ensure all langchain packages are installed in the venv.
- Possibly add error handling for missing packages.
- Consider adding CORS middleware (commented out currently).

---

This document will be updated as the project evolves.