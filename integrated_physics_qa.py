# Integrated Physics QA System
# This script combines the image processing and QA system while preserving original code

# Import all necessary libraries
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai

# Image processing functions (from image_090909.py)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def enhance_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def post_process_physics_text(text):
    physics_replacements = {
        r'(\d+)\s*ms?\^?-?1\b': r'\1 m/s',  # Fix meters per second
        r'(\d+)\s*m/s\^?2\b': r'\1 m/s²',   # Fix acceleration units
        r'\bAp\b': 'Δp',                    # Change in momentum
        r'\bdv\b': 'Δv',                    # Change in velocity
        r'\bdt\b': 'Δt',                    # Change in time
        r'(\d+)\s*([NJ])\b': r'\1 \2',      # Space between number and N or J
        r'(\d+)\s*kg\b': r'\1 kg',          # Space before kg
        r'(\d+)\s*m\b': r'\1 m',            # Space before m (meters)
        r'(\d+)\s*s\b': r'\1 s',            # Space before s (seconds)
        r'(\d+)\s*K\b': r'\1 K',            # Space before K (Kelvin)
        r'(\d+)\s*Pa\b': r'\1 Pa',          # Space before Pa (Pascal)
        r'(\d+)\s*W\b': r'\1 W',            # Space before W (Watt)
        r'\bF\s*=\s*ma\b': 'F = ma',        # Newton's Second Law
        r'\bE\s*=\s*mc\^?2\b': 'E = mc²',   # Einstein's mass-energy equivalence
        r'\bPV\s*=\s*nRT\b': 'PV = nRT',    # Ideal Gas Law
        r'\bv\^?2\b': 'v²',                 # Velocity squared
        r'\ba\^?2\b': 'a²',                 # Acceleration squared
        r'\bπ': 'π',                        # Ensure pi symbol is correct
        r'\b([pv])1\b': r'\1₁',             # Subscript 1 for p or v
        r'\b([pv])2\b': r'\1₂',             # Subscript 2 for p or v
        r'(\d+)([,.])(\d+)': r'\1.\3',      # Standardize decimal point
    }

    for pattern, replacement in physics_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text

def ocr_physics_question(image_path):
    image = cv2.imread(image_path)
    preprocessed = preprocess_image(image)
    enhanced = enhance_image(preprocessed)

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-+=^°ΔπμρλσωΩ "'

    text = pytesseract.image_to_string(enhanced, config=custom_config)
    corrected_text = post_process_physics_text(text)

    return corrected_text

# QA System functions (from physics_fyp (1).py)

def read_pdf(file_path: str = "phys-tbk.pdf") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pdf_reader = PdfReader(file_path)
    content = ""
    for page in tqdm(pdf_reader.pages, desc="Reading PDF"):
        content += page.extract_text()

    print(f"PDF processed. Total characters: {len(content)}")
    return content

def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks")
    if chunks:
        print(f"First chunk length: {len(chunks[0])}")
        print(f"First chunk preview: {chunks[0][:100]}...")
        print(f"Last chunk preview: {chunks[-1][-100:]}...")
    else:
        print("Warning: No chunks created!")
    return chunks

def create_embeddings(chunks: list) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks, show_progress_bar=True)

def save_data(chunks: list, embeddings: np.ndarray, save_path: str):
    np.savez_compressed(save_path, chunks=chunks, embeddings=embeddings)
    print(f"Data saved to {save_path}")

def preprocess_dataset(dataset, save_path="physics_dataset_embeddings.npz"):
    if os.path.exists(save_path):
        print("Loading pre-processed dataset...")
        data = np.load(save_path, allow_pickle=True)
        return data['texts'], data['embeddings']

    print("Processing dataset...")
    texts = [f"{item['instruction']} {item['output']}" for item in dataset]
    embeddings = create_embeddings(texts)
    save_data(texts, embeddings, save_path)
    return texts, embeddings

def process_textbook(file_path: str = "phys-tbk.pdf", save_path: str = "physics_textbook_data.npz"):
    content = read_pdf(file_path)
    chunks = create_chunks(content)
    if not chunks:
        raise ValueError("No chunks created. Check the text splitting process.")
    embeddings = create_embeddings(chunks)
    save_data(chunks, embeddings, save_path)
    return chunks, embeddings

def load_physics_dataset():
    dataset = load_dataset("Akul/alpaca_physics_dataset")
    return dataset['train']

def setup_qa_system(textbook_chunks, textbook_embeddings, dataset_texts, dataset_embeddings):
    all_texts = textbook_chunks + dataset_texts
    all_embeddings = np.vstack([textbook_embeddings, dataset_embeddings])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings.tolist())),
        embedding=embeddings,
        metadatas=[{"source": "textbook"} if i < len(textbook_chunks) else {"source": "dataset"}
                   for i in range(len(all_texts))]
    )

    return db.as_retriever(search_kwargs={"k": 5})

def generate_calculation_flowchart(answer):
    print("Generating flowchart...")
    print(f"Answer received: {answer}")

    # Extract steps from the answer
    steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', answer, re.DOTALL)
    print(f"Steps extracted: {steps}")

    # Generate Mermaid flowchart
    mermaid_code = "graph TD\n"
    for i, step in enumerate(steps):
        node_id = f"A{i}"
        next_node_id = f"A{i+1}"

        # Clean up the step text
        step = step.strip().replace("\n", "<br>")

        # Add node to flowchart
        mermaid_code += f"    {node_id}[{step}]\n"

        # Add connection to next step
        if i < len(steps) - 1:
            mermaid_code += f"    {node_id} --> {next_node_id}\n"

    print(f"Generated Mermaid code: {mermaid_code}")
    return mermaid_code

def ask_question(retriever, question, conversation_history):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel('gemini-pro')

    assistant_prompt = """
    You are an expert physics tutor explaining a problem to a student. Answer the question in a clear, step-by-step manner that can be easily translated into a flowchart. Follow these guidelines:

    1. Pay attention to the conversation history and previous questions to maintain context.
    2. If the question is a follow-up or relates to previous questions, use the information from earlier in the conversation.
    3. For calculation questions:
       - Clearly state the given information, including any from previous questions if relevant.
       - List the relevant formulas needed to solve the problem.
       - Explain the solution process step-by-step, showing all work and reasoning.
       - Provide the final answer with appropriate units.
    4. Use clear, numbered steps that can be easily translated into a flowchart.
    5. If asked to calculate something specific, provide the calculation without asking for more information unless absolutely necessary.

    Adapt your response style to the nature of the question, providing a clear, structured explanation that follows a logical problem-solving flow.
    """

    conversation_context = "\n".join(conversation_history)
    prompt = f"{assistant_prompt}\n\nConversation history:\n{conversation_context}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = model.generate_content(prompt)
    answer = response.text

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Check if the question involves calculations
    if any(keyword in question.lower() for keyword in ['calculate', 'compute', 'find the value']):
        print("Calculation detected, generating flowchart...")
        flowchart = generate_calculation_flowchart(answer)
        print("\nHere's a flowchart of the calculation process:")
        print(flowchart)
    else:
        print("No calculation detected, skipping flowchart generation.")
        flowchart = None

    return answer, flowchart

# Main execution
def main():
    print("Initializing Physics QA System...")
    genai.configure(api_key="AIzaSyCnMQavoRz3ZTojwFfHg66wrmHmNv_qHC4")  # Replace with your actual API key

    # Process textbook
    textbook_chunks, textbook_embeddings = process_textbook()

    # Load and process dataset
    dataset = load_physics_dataset()
    dataset_texts, dataset_embeddings = preprocess_dataset(dataset)

    # Set up the QA system
    retriever = setup_qa_system(textbook_chunks, textbook_embeddings, dataset_texts, dataset_embeddings)

    conversation_history = []
    while True:
        user_input = input("Enter your physics question, type 'image' to upload an image, or 'quit' to exit: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Physics QA system. Goodbye!")
            break
        
        if user_input.lower() == 'image':
            image_path = input("Enter the path to your image file: ").strip()
            if os.path.exists(image_path):
                question = ocr_physics_question(image_path)
                print(f"Extracted question from image: {question}")
            else:
                print("Image file not found. Please try again.")
                continue
        else:
            question = user_input

        try:
            answer, flowchart = ask_question(retriever, question, conversation_history)
            print(f"\nAnswer: {answer}\n")
            if flowchart:
                print("Calculation flowchart:")
                print(flowchart)
            conversation_history.append(f"Q: {question}")
            conversation_history.append(f"A: {answer}")
        except Exception as e:
            print(f"An error occurred while answering the question: {str(e)}")

if __name__ == "__main__":
    main()