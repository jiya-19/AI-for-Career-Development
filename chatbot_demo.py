import requests

API_URL = "http://127.0.0.1:5000/get_answer"  # Ensure the API is running at this address
SMALLTALK_API_URL = "http://127.0.0.1:5000/get_smalltalk"  # Ensure the API is running at this address

def ask_chatbot(question):
    response = requests.post(API_URL, json={"question": question})
    if response.status_code == 200:
        data = response.json()
        return f"\nChatbot: {data['answer']}\n"
    else:
        return "\nChatbot: Error fetching response."

print("Interview Chatbot (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("\nChatbot: Goodbye!")
        break
    print(ask_chatbot(user_input))
