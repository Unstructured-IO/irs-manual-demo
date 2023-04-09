## Chat with IRS Manuals

This directory contains an application for chatting with IRS manuals. The chat application only uses self-hosted models and can be run in a disconnected environment. Here's how to get started with the chatbot:

- Pip install requirements.txt
- Run `download_data.py` in order to grab all the pdfs
- Run pdfs against unstructured
- `PYTHONPATH=. ./unstructured/ingest/main.py   --local-input-path ../zips --structured-output-dir ../zips/output`
- Run `ingest_data.py` to push content to pinecone
- Run `python cli.py` to chat via a command line interface

# TODO:
- add gifs to readme showing steps
  - downloading/ingesting
  - running through unstructured (use api instead)
  - using cli
- more detailed install instructions
- put on huggingface?
- add some prompt questions
- move api keys into env vars
- push pdfs/jsons to repo
