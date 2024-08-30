from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Carregar o modelo e o tokenizer do DialoGPT
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurar o pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=150, truncation=True)

def chat_with_bot():
    print("Chatbot: Olá! Como posso ajudar você hoje? (Digite 'sair' para encerrar a conversa)")
    chat_history = []

    while True:
        user_input = input("Você: ")
        
        if user_input.lower() in ["sair", "exit"]:
            print("Chatbot: Até logo!")
            break
        
        # Adiciona a entrada do usuário ao histórico
        chat_history.append(f"Você: {user_input}")

        # Cria o contexto para o modelo
        chat_history_str = "\n".join(chat_history) + "\nChatbot:"

        # Gera a resposta do chatbot
        response = chatbot(chat_history_str, pad_token_id=tokenizer.eos_token_id, max_length=200, truncation=True)

        # Extrai a resposta gerada
        generated_text = response[0]['generated_text']

        # Remove o texto de entrada do histórico
        response_text = generated_text.split("Chatbot:")[-1].strip()

        # Adiciona a resposta do chatbot ao histórico
        chat_history.append(f"Chatbot: {response_text}")

        # Exibe a resposta do chatbot
        print(f"Chatbot: {response_text}")

# Iniciar a função de chat
chat_with_bot()