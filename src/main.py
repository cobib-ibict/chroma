'''
Copyright (c) 2025 Ibict Authors. All rights reserved.
'''



'''
To load parameters from .env, add the following content to the bottom of your `venv/bin/activate` (`nano venv/bin/activate`):
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi



Future update:
- Langchain to produce embeddings (OpenAI charges per million tokens, so using local models is preferred for now)
'''

import abc
import asyncio
import chromadb
import dotenv
import ijson
import json
import openai
import os
import pathlib
import re
import time
# import traceback # Debugging (t.print_stack())import os

# Read .env file content
dotenv.load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Constant
JSON_PARSING_PRINT_CYCLETIME    = 1 # seconds
VECTORDB_SAVING_PRINT_CYCLETIME = 1 # seconds

# Error code
RAG_ERROR                  = 0
RAG_ERROR_NOJSONFILEPATH   = 1
RAG_ERROR_NOCLIENT         = 2
RAG_ERROR_NOCOLLECTION     = 3
RAG_ERROR_NOCOLLECTIONNAME = 4

# OpenAI
client_ai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))





# region MARK: RagHandler

class RagHandler(abc.ABC):
  # Error message
  __ERROR_MESSAGE = {
    RAG_ERROR                 : "Sorry, not possible.",
    RAG_ERROR_NOJSONFILEPATH  : "JSON file path is not provided.",
    RAG_ERROR_NOCLIENT        : "Client is not created yet.",
    RAG_ERROR_NOCOLLECTION    : "Collection is not created yet.",
    RAG_ERROR_NOCOLLECTIONNAME: "Collection name is not provided."
  }



  # region MARK:.  Self

  def __init__(self, json_filepath: str = None, collection_name: str = None):
    '''
    Constructor
    '''

    # Parameter
    self.json_filepath   = json_filepath
    self.collection_name = collection_name

    # JSON parsing data
    self.unique_ids = [ ]
    self.metadatas  = [ ]
    self.documents  = [ ]

    # Vector database
    self.client     = None
    self.collection = None

  @staticmethod
  def error(id) -> str:
    '''
    Get error message.

    Args:
      id (int): Error id.

    Returns:
      str: Error message.
    '''

    return RagHandler.__ERROR_MESSAGE[id]

  def has_data(self) -> bool:
    '''
    Check if collection has data.

    Returns:
      bool: `True` if collection has data, `False` otherwise.
    '''

    return self.collection and self.collection.count()

  # endregion



  # region MARK:.  JSON parsing

  def __parse_object(self, obj: dict) -> tuple:
    '''
    Parse a single object from a JSON file.

    Args:
      obj (dict): A single object from a JSON file.

    Returns:
      tuple: A tuple of metadatas and documents.
    '''

    # Note:
    # There is a problem that obj.get('INSTITUICAO', '') only uses the default value if the key doesn't exist in json.
    # But if the key explicitly exists with a `null` value (which becomes `None` in Python), it returns the explicit `None`, instead of the default value.
    # One work around for this, is to force the fallback manually. For example: `obj.get('INSTITUICAO') or ''`.
    # Therefore, if the value is `None`, `''` or another "falsy" value, then `''` will be used instead.

    # Metadatas
    metadatas = {
      'COD_CCN_PUBLICACAO': (obj.get('COD_CCN_PUBLICACAO') or '').strip(),
      'TITULO_PUBLICACAO' : (obj.get('TITULO_PUBLICACAO') or '').strip(),
      'TITULO_RELACIONADO': (obj.get('TITULO_RELACIONADO') or '').strip(),
      'BIBLIOTECA_NOME'   : (obj.get('BIBLIOTECA_NOME') or '').strip(),
      'INSTITUICAO'       : (obj.get('INSTITUICAO') or '').strip(),
      'NOME_EDITORA'      : (obj.get('NOME_EDITORA') or '').strip(),
      'AREA_CONHECIMENTO' : (obj.get('AREA_CONHECIMENTO') or '').strip(),
      'SPINES'            : (obj.get('SPINES') or '').strip(),
      'TERMO_LIVRE'       : (obj.get('TERMO_LIVRE') or '').strip(),
      'COLECAO'           : (obj.get('COLECAO') or '').strip(),
      'COMENTARIO'        : (obj.get('COMENTARIO') or '').strip(),
    }

    # Document
    doc = f"Código CCN: {metadatas['COD_CCN_PUBLICACAO']}\n\n" \
          f"Título: {metadatas['TITULO_PUBLICACAO']}\n\n" \
          f"Título relacionado: {metadatas['TITULO_RELACIONADO']}\n\n" \
          f"Biblioteca: {metadatas['BIBLIOTECA_NOME']}\n\n" \
          f"Instituição: {metadatas['INSTITUICAO']}\n\n" \
          f"Editora: {metadatas['NOME_EDITORA']}\n\n" \
          f"Área do conhecimento: {metadatas['AREA_CONHECIMENTO']}\n\n" \
          f"Assuntos controlados: {metadatas['SPINES']}\n\n" \
          f"Termo livre: {metadatas['TERMO_LIVRE']}\n\n" \
          f"Coleção: {metadatas['COLECAO']}\n\n" \
          f"Comentário: {metadatas['COMENTARIO']}"

    return metadatas, doc

  def __parse_json_file(self, limit: int = None):
    '''
    Parse a JSON file in streaming mode (reads without loading entire file into memory).
    It also yields (produces) documents to the generator.

    Args:
      limit (int, optional): The maximum number of objects to parse.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    global JSON_PARSING_PRINT_CYCLETIME

    parsed_amount = 0
    json_info     = self.get_json_file_info()
    time_begin    = time.time()
    time_previous = time_begin # Time of previous batch execution

    print(f">> Starting to parse '{json_info[0].name}' ({json_info[1]:,.2f} MB)...")

    with open(self.json_filepath, 'rb') as file:
      # Parse streaming

      '''
      Expected json format:
      [
        { ... }, # Item 1
        { ... }, # Item 2
        { ... }  # Item 3
      ]
      '''

      # Reset file cursor
      file.seek(0)

      for obj in ijson.items(file, prefix = 'item'):
        parsed_amount += 1 # Used as unique id
        data          = self.__parse_object(obj)
        metadatas     = data[0]
        doc           = data[1]

        # Deliver id, metadatas and document to the caller (generator)
        yield parsed_amount, metadatas, doc

        # Display progress
        time_current = time.time() # Time of current batch execution
        if time_current - time_previous > JSON_PARSING_PRINT_CYCLETIME:
          time_previous = time_current

          # Print actual progress
          time_elapsed = round(time_current - time_begin)
          print(f"> {parsed_amount:,} objects parsed in {time_elapsed} second{'' if time_elapsed < 2 else 's'}.")

        # Stop if limit is reached
        if limit and parsed_amount >= limit:
          break

    time_current = time_current or time.time() # Time of current batch execution
    print(f"> {parsed_amount:,} objects parsed from '{json_info[0].name}' ({json_info[1]:,.2f} MB) in {round(time_current - time_begin, 2):,.2f} seconds.\n")

  def load(self, *a, **k):
    '''
    Parse a JSON file and store its data in internal lists.
    Same parameters as __parse_json_file.

    Args:
      *a (tuple): Positional arguments.
      **k (dict): Keyword arguments.
    '''

    for id, metadatas, doc in self.__parse_json_file(*a, **k):
      # Store unique id, metadatas and document
      self.unique_ids.append(str(id))
      self.metadatas.append(metadatas)
      self.documents.append(doc)

  def get_json_file_info(self) -> tuple:
    '''
    Get information about a JSON file.

    Returns:
      path (pathlib.Path): The path to the JSON file.
      mb_size (float): The size of the JSON file in megabytes.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    path    = pathlib.Path(self.json_filepath)
    mb_size = round(path.stat().st_size / (1024 ** 2), 2)

    return path, mb_size

  def print_json_file_data(self, *a, **k):
    '''
    Parse a JSON file and print its data.
    Same parameters as __parse_json_file.

    Args:
      *a (tuple): Positional arguments.
      **k (dict): Keyword arguments.
    '''

    for id, metadatas, doc in self.__parse_json_file(*a, **k):
      print(f">> Found new document:\n")

      print(f"> Metadatas:\n{metadatas}\n")
      print(f"> Document #{id}:\n{doc}\n")

      print('\n' + ('--- ' * 7) + '\n')

  # endregion



  # region MARK:.  Chroma

  # region MARK:.    VectorDB

  @abc.abstractmethod
  def create_client(self):
    '''
    Create a Chroma client.
    '''

    pass

  @abc.abstractmethod
  def create_collection(self):
    '''
    Create a Chroma collection.
    '''

    pass

  @abc.abstractmethod
  def delete_collection(self):
    '''
    Delete the Chroma collection.
    '''

    pass

  # endregion

  # endregion

# endregion



# region MARK: PersistentRagHandler

class PersistentRagHandler(RagHandler):
  # region MARK:.  Self

  def __init__(self, json_filepath: str = None, client_path: str = None, collection_name: str = None, embedding_function: object = None):
    # Call the inherited constructor
    super().__init__(json_filepath = json_filepath, collection_name = collection_name)

    # Parameter
    self.client_path        = client_path
    self.embedding_function = embedding_function

    # Initialize
    self.init()

  def init(self):
    # Attempt to init
    if self.client_path:
      self.create_client()
    if self.collection_name:
      self.create_collection()

    return self

  # endregion



  # region MARK:.  Chroma

  # region MARK:.    VectorDB

  def create_client(self):
    print(">> Creating client...")

    # Create a Chroma persistent client
    self.client = chromadb.PersistentClient(path = self.client_path, settings = chromadb.config.Settings(anonymized_telemetry = False))

    print(f"> Client has been created successfully with path as '{self.client_path}'.\n")

  def create_collection(self):
    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)
    assert self.collection_name, RagHandler.error(RAG_ERROR_NOCOLLECTIONNAME)

    print(">> Creating collection...")

    self.collection = self.client.get_or_create_collection(name = self.collection_name, embedding_function = self.embedding_function)
    amount          = self.collection.count()
    amount_txt      = f" with {amount:,} items" if amount else ''

    print(f"> Collection '{self.collection_name}' has been created successfully{amount_txt}.\n")

  def delete_collection(self):
    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    for collection in self.client.list_collections():
      try:
        self.client.delete_collection(name = collection.name)
      except:
        pass

  def create_vectordb(self):
    '''
    Create a Chroma vector database.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)
    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    global VECTORDB_SAVING_PRINT_CYCLETIME

    batch_range   = 100 # Number of objects to add at once per batch
    saved_amount  = 0
    total_amount  = len(self.unique_ids)
    time_begin    = time.time()
    time_previous = time_begin # Time of previous batch execution
    time_current  = time_begin # Time of current batch execution

    print(f">> Starting to create the vector database...")

    # If empty, stop
    if total_amount == 0:
      print("> Nothing to save.")
      return

    for i in range(0, total_amount, batch_range):
      index_end = min(i + batch_range, total_amount)

      # Add content to the collection
      try:
        # .upsert instead of .add to avoid adding the same documents every time
        self.collection.upsert(ids = self.unique_ids[i:index_end], metadatas = self.metadatas[i:index_end], documents = self.documents[i:index_end])

        saved_amount += index_end - i
      except Exception as err:
        print(f"> Error in batch from {i} to {index_end}: {err}")

      # Batch progress

      time_current = time.time()
      if time_current - time_previous > VECTORDB_SAVING_PRINT_CYCLETIME:
        time_previous  = time_current
        progress_ratio = (saved_amount / total_amount) if total_amount else 0

        # Complete already, then break
        if progress_ratio >= 1:
          break

        # Print actual progress
        time_elapsed = round(time_current - time_begin)
        print(f"> Saved {saved_amount:,} documents in {time_elapsed} second{'' if time_elapsed < 2 else 's'} ({(progress_ratio * 100):.2f}%).")

    # Print last progress
    print(f"> Saved {saved_amount:,} documents in {round(time_current - time_begin, 2):,.2f} seconds.\n")

  # endregion



  # region MARK:.    Search

  def search(self, query_text: str, n_results: int = 10):
    '''
    Search for relevant documents in the Chroma collection.
    It needs a vector database to be already created and filled with the embeddings of the documents.

    Args:
      query_text (str): Query text to search for.
      n_results (int, optional): Number of results.
    '''

    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    print(f"\n>> Requested query: \"{query_text}\"")

    # Search for relevant documents and their metadatas
    results = self.collection.query(query_texts = [query_text], n_results = n_results)

    # Get the metadatas and documents
    ids       = results['ids'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    # Display the results

    # Found results
    if ids and metadatas and documents:
      print(f">> Found {len(documents)} relevant results:\n")

      for i, (id, metadata, document) in enumerate(zip(ids, metadatas, documents), 1):
        print(f"> Result #{i} - Document #{id}\n")
        print(f"> Metadatas:\n{metadata}\n")
        print(f"> Document:\n{document}\n")
        print('\n' + ('--- ' * 7) + '\n')

    # No results found
    else:
      print(">> No results found.")

  def init_search_terminal_mode(self, n_results: int = 10):
    '''
    It uses the search method to search in the terminal for relevant documents in the Chroma collection.

    Args:
      n_results (int, optional): Number of results.
    '''

    print(">> RAG Search")

    while True:
      try:
        query = input("\nType your question (use \"Ctrl+C\" or type \"quit\" to exit): ").strip()

        if not query:
          print(">> Type a valid question.")
          continue

        if query.lower() in ['quit', 'q', 'exit', 'sair']:
          raise KeyboardInterrupt

        self.search(query_text = query, n_results = n_results)

      except KeyboardInterrupt:
        print("\n>> Ending session...")
        break

  # endregion

  # endregion

# endregion



# region MARK: AsyncHttpRagHandler

class AsyncHttpRagHandler(RagHandler):
  # region MARK:.  Self

  def __init__(self, json_filepath: str = None, client_host: str = 'localhost', client_port: int = 8000, collection_name: str = None):
    # Call the inherited constructor
    super().__init__(json_filepath = json_filepath, collection_name = collection_name)

    # Parameter
    self.client_host = client_host
    self.client_port = client_port

    # Prepare initialization (not executed yet)
    self._init_coroutine = self.init()

  async def init(self):
    # Attempt to init
    if self.client_host and self.client_port:
      await self.create_client()
    if self.collection_name:
      await self.create_collection()

    return self

  def __await__(self):
    return self._init_coroutine.__await__() # It makes `await AsyncHttpRagHandler(...)` to be possible

  # endregion



  # region MARK:.  Chroma

  # region MARK:.    VectorDB

  async def create_client(self):
    print(">> Creating client...")

    # Create a Chroma async HTTP client
    self.client = await chromadb.AsyncHttpClient(host = self.client_host, port = self.client_port, settings = chromadb.config.Settings(anonymized_telemetry = False))

    print(f"> Client has been created successfully with host as '{self.client_host}' and port as '{self.client_port}'.\n")

  async def create_collection(self):
    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)
    assert self.collection_name, RagHandler.error(RAG_ERROR_NOCOLLECTIONNAME)

    print(">> Creating collection...")

    self.collection = await self.client.get_or_create_collection(name = self.collection_name)
    amount          = await self.collection.count()
    amount_txt      = f" with {amount:,} items" if amount else ''

    print(f"> Collection '{self.collection_name}' has been created successfully{amount_txt}.\n")

  async def delete_collection(self):
    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    for collection in self.client.list_collections():
      try:
        await self.client.delete_collection(name = collection.name)
      except:
        pass

  # endregion



  # region MARK:.    Search

  # endregion

  # endregion

# endregion



# region MARK: OpenAI

# Case 1 of OpenAI request
def search_colecao(data: dict):
  '''
  Search in the API for the collection.
  It starts by searching for the title of "publicação seriada" or "biblioteca", getting its id.
  And then by searching for the collection with that id.

  Args:
    titulo (str): The title of the "publicação seriada" or "biblioteca".
    biblioteca (bool, optional): If the collection is a "biblioteca" (True) or "publicação seriada" (False).
  '''

  print(f"\n>> Busca via API para COLECAO")

  publicacao = (data.get('publicacao') or '').strip()
  biblioteca = (data.get('biblioteca') or '').strip()

  if publicacao:
    print(f"> Publicação seriada: {publicacao}")

  elif biblioteca:
    print(f"> Biblioteca: {biblioteca}")

  else:
    print(f"Não foi possível executar a busca com os parâmetros fornecidos:\n{data}")

def get_json_from_text(text: str):
  '''
  Get the JSON from the text.

  Args:
    text (str): The text to get the JSON from.

  Returns:
    dict: The JSON data.
  '''

  # Slice the JSON from the string
  json_slice = re.search(r"\{[\s\S]*\}", text)
  json_slice = json_slice.group(0) if json_slice else None

  # Transform it to JSON
  try:
    json_data = json.loads(json_slice) if json_slice else None
  except json.JSONDecodeError:
    json_data = None

  return json_data

# Init the AI terminal
def init_ai_terminal(rag_search: PersistentRagHandler, n_results: int = 10):
  print(">> RAG Search")

  while True:
    try:
      query = input("\nType your question (use \"Ctrl+C\" or type \"quit\" to exit): ").strip()

      # Initialize function variables
      func      = None
      func_args = None

      if not query:
        print(">> Type a valid question.")
        continue

      if query.lower() in ['quit', 'q', 'exit', 'sair']:
        raise KeyboardInterrupt

      # Call OpenAI
      response = client_ai.responses.create(
        model = 'gpt-4o-mini',
        input = (
          'Você é um roteador semântico que só responde em formato JSON com as informações requisitadas.\n\n'
          f'"Requisição do usuário": {query}\n\n'
          '* Caso 1:\n'
          'Preciso de 2 informações extraídas da "requisição do usuário".\n'
          'Quando a "requisição do usuário" mencionar "coleções de determinada publicação seriada" (ou revista/jornal/obra/etc), ou "coleções de determinada biblioteca", identifique o "título da publicação seriada" ou o "título da biblioteca".\n'
          'O "nome da função" para este caso é sempre "search_colecao".'
          'A sua resposta deve estar sempre no seguinte formato JSON (não quero nada além disso):\n'
          '```json\n{ "func": "nome da função", "publicacao": "título da publicação" }\n```\n'
          'ou\n'
          '```json\n{ "func": "nome da função", "biblioteca": "título da biblioteca" }\n```\n\n'
          '* Caso final:'
          'Em caso de não identificar nenhum caso acima, sua resposta deve ser:'
          '```json\n{ }\n```\n\n'
          'Notas importantes:\n'
          '* Se algum valor do JSON estiver sintaticamente errado ou faltando acentuação, exceto pelo valor da "func" (obviamente), corrija se tiver certeza de como é o correto (caso contrário, mantenha como está). Por exemplo, "Cieência da computacao" seria corrigo por "Ciência da Computação".'
        ),
      )

      # Get output (which has always length 1, for our model)
      response_output = response.output[0]

      # print(f"> AI output: {response_output}\n") # -- For debug --

      # Received a message response
      if response_output.type == 'message':

        # print(f"> Response content: {response_output.content}\n") # -- For debug --

        response_text = response_output.content[0].text

        # print(f"> Response text: {response_text}\n") # -- For debug --

        # Get the JSON from the text
        json_data = get_json_from_text(response_text)

        # print(f"> JSON data: {json.dumps(json_data, indent = 2, ensure_ascii = False) if json_data else None}\n") # -- For debug --

        if json_data and json_data.get('func'):

          # Get the function name
          func_name = json_data.get('func')

          # Get the function arguments
          func_args = { key: value for key, value in json_data.items() if key != 'func' }

          # print(f"> Function name: {func_name}\n") # -- For debug --
          # print(f"> Function arguments: {json.dumps(func_args, indent = 2, ensure_ascii = False)}\n") # -- For debug --

          # Get the function
          func = globals().get(func_name)

          # print(f"> Function: {func}\n") # -- For debug --

      # Call the function
      if func:
        func(func_args) # Todo: do something with the results of this function

      # Default: RAG search
      else:
        rag_search.search(query, n_results=n_results)

    except KeyboardInterrupt:
      print("\n>> Ending session...")
      break

# endregion





# region MARK: Main

async def main():
  '''
  Parse a JSON file and print its data
  '''

  # # Get info about a JSON file
  # rag_parser = PersistentRagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # print(rag_parser.get_json_file_info())

  # # Parse a JSON file in streaming mode
  # rag_parser = PersistentRagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # rag_parser.print_json_file_data(limit = 10)

  # # Parse a JSON file and print its data of internal lists
  # rag_parser = PersistentRagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # rag_parser.load(limit = 10)
  # print()
  # print(rag_parser.unique_ids)
  # print()
  # print(rag_parser.metadatas)
  # print()
  # print(rag_parser.documents)



  '''
  Parse a JSON file into a vector database
  '''

  # # Parse a JSON file and save its data as a vector database
  # print(">> Creating embedding function...")
  # embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')
  # print("> Embedding function has been created successfully.\n")
  # rag_vectordb = PersistentRagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json', client_path = f'{PROJECT_ROOT}/output', collection_name = 'data', embedding_function = embedding_function)
  # rag_vectordb.load(limit = 250)
  # rag_vectordb.create_vectordb()



  '''
  Search in a vector database
  '''

  # # Search in the vector database
  # print(">> Creating embedding function...")
  # embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')
  # print("> Embedding function has been created successfully.\n")
  # rag_search = PersistentRagHandler(client_path = f'{PROJECT_ROOT}/output', collection_name = 'data', embedding_function = embedding_function)
  # rag_search.search(query_text = "Me mostre publicações de psicologia da Universidade Federal de Pernambuco", n_results = 10)

  # # Init search in terminal mode
  # print(">> Creating embedding function...")
  # embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')
  # print("> Embedding function has been created successfully.\n")
  # rag_search = PersistentRagHandler(client_path = f'{PROJECT_ROOT}/output', collection_name = 'data', embedding_function = embedding_function)
  # rag_search.init_search_terminal_mode(n_results = 10)



  '''
  Search API using OpenAI handling or in a vector database
  '''

  # Init search in AI terminal mode
  print(">> Creating embedding function...")
  embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')
  print("> Embedding function has been created successfully.\n")
  rag_search = PersistentRagHandler(client_path = f'{PROJECT_ROOT}/output', collection_name = 'data', embedding_function = embedding_function)
  init_ai_terminal(rag_search, n_results = 3)



  '''
  Search in a vector database via HTTP (needs the Chroma server running: `chroma run --path ./output`)
  '''

  # # Todo
  # rag_http_search = await AsyncHttpRagHandler(client_host = 'localhost', client_port = 8000, collection_name = 'data')
  # print(rag_http_search.client)
  # print(rag_http_search.collection)
  #



if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    pass

# endregion
