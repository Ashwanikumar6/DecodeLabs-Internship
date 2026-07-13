import time
import random

responses = {
    "what is your name": "My name is DecodeBot.",
    "how are you": "I am doing great! What about you?",
    "i am fine": "That's great to hear! 😊",
    "who made you": "I was created by Ashwani Kumar.",
    "what is ai": "AI stands for Artificial Intelligence.",
    "what is your work": "I am a rule-based AI chatbot. I answer predefined questions and assist users.",
    "what can you do": "I can answer basic predefined questions, tell the time, date and day.",
    "thank you": "You're welcome! 😊",
    "thanks": "You're welcome! 😊"
}

greetings = [
    "Hello! How can I help you today?",
    "Hi! Nice to meet you.",
    "Welcome! What would you like to know?",
    "Hey! How can I assist you?"
]


def chatbot():

    print("=" * 50)
    print("Welcome to DecodeBot")
    print("Type 'help' to see available commands.")
    print("Type 'exit' anytime to quit.")
    print("=" * 50)

    while True:

        user_input = input("\nYou : ")

        clean_user_input = (
            user_input.lower()
            .strip()
            .replace("?", "")
            .replace(".", "")
            .replace(",", "")
            .replace("!", "")
        )

        if clean_user_input in ["exit", "quit", "bye"]:
            print("DecodeBot : Goodbye! Have a wonderful day. 👋")
            break

        elif clean_user_input in [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "good afternoon",
            "what's up"
        ]:
            print("DecodeBot :", random.choice(greetings))

        elif clean_user_input in responses:
            print("DecodeBot :", responses[clean_user_input])

        elif "python" in clean_user_input:
            print("DecodeBot : Python is a high-level, easy-to-learn and versatile programming language.")

        elif "time" in clean_user_input:
            print("DecodeBot : Current time is", time.strftime("%H:%M:%S"))

        elif "date" in clean_user_input:
            print("DecodeBot : Today's date is", time.strftime("%d/%m/%Y"))

        elif "day" in clean_user_input:
            print("DecodeBot : Today is", time.strftime("%A"))

        elif clean_user_input == "help":
            print("\nI can answer questions like:")
            print("- Hi / Hello")
            print("- What is your name")
            print("- How are you")
            print("- Who made you")
            print("- What is AI")
            print("- What is Python")
            print("- What is your work")
            print("- What can you do")
            print("- What time is it")
            print("- What's today's date")
            print("- What day is today")
            print("- Thank you")
            print("- Exit")

        else:
            print("DecodeBot : Sorry, I don't know the answer to that. Please ask another question.")


def main():
    chatbot()


if __name__ == "__main__":
    main()