import argparse
from ai_agent.ai_agent import AIAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RAG Agent for PDF processing and Q&A")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("questions_file", help="Path to the file containing questions, one per line")
    
    args = parser.parse_args()
    
    # Create AI Agent
    ai_agent = AIAgent(args.pdf_path)
    with open(args.questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    # Process PDF and answer questions
    ai_agent.run(questions)

if __name__ == "__main__":
    main()