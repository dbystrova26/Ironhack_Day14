Create a RAG that is capable of answering questions about a sport in an accurate and accessible manner for beginners.

Steps

Imports details:
Import our .env file with load_dot_env
Use langchain 1.0 and compatible libraries (langchain_openai, langchain_pinecone)
We will process a pdf (library to be decided)

Process the PDF
Keep each section together (no chunking)
Metadata should contain: name, page, section, section_name, previous_section_id, foward_section_id
For the images in the pdf, call open ai to provide a detail description of the image and embed that description. Add the image to the metadata.
Embed the chunk
embed small from openai
Upsert them in pinecone
index_name = 'sports_rules'