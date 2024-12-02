from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.qdrant import Qdrant
import qdrant_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from Template.promptAI import AI_prompt
import azure.cognitiveservices.speech as speechsdk
import base64
import os
import orjson

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Allow CORS for specific origin

# Azure Speech SDK Configuration
speech_key = os.getenv("SPEECH_KEY")
service_region = os.getenv("SPEECH_REGION")

# Qdrant Vector Store Configuration
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# Global chat history
chat_history = []

# Function to initialize vector store
def get_vector_store():
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store
vector_store = get_vector_store()
def get_context_retriever_chain(vector_store=vector_store):
    llm = ChatOpenAI(model='gpt-4o')
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query based on the conversation."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Function to create the conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model='gpt-4o')
    prompt = ChatPromptTemplate.from_messages([
        ("system", AI_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Initialize Azure Speech Synthesizer
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "en-GB-AdaMultilingualNeural"
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# TTS function to synthesize response and generate viseme data
def synthesize_response(text):
    visemes = []

    def viseme_callback(evt):
        visemes.append([evt.audio_offset / 10000, evt.viseme_id])

    synthesizer.viseme_received.connect(viseme_callback)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = base64.b64encode(result.audio_data).decode("utf-8")
        return {
            "audio": audio_data,
            "visemes": visemes,
        }
    else:
        return {"audio": None, "visemes": [], "error": "Synthesis failed"}

# Efficient JSON response helper
def jsonify_fast(data):
    return app.response_class(response=orjson.dumps(data), mimetype="application/json")

# RAG endpoint
@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')

    global chat_history
    if chat_history is None:
        chat_history = []

    chat_history.append(HumanMessage(content=user_input))

    try:
        # Initialize vector store dynamically
        vector_store = get_vector_store()

        # Generate context retriever chain dynamically
        retriever_chain = get_context_retriever_chain(vector_store)

        # Generate conversational RAG chain dynamically
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        # Generate response
        response = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
        })

        response_content = response.get("answer", "")
        chat_history.append(AIMessage(content=response_content))

        # Generate audio and viseme data
        audio_response = synthesize_response(response_content)

        return jsonify_fast({
            "response": response_content,
            "audio": audio_response.get("audio"),
            "visemes": audio_response.get("visemes"),
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify_fast({"error": "An error occurred while generating the response."}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True, threaded=True)






