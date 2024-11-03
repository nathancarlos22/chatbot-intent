# Chatbot de Reconhecimento de Intenções

Este projeto é um chatbot que utiliza o modelo BERT para identificar intenções de usuários em entradas de texto e responder de forma personalizada. Ele é treinado para reconhecer até 10 intenções diferentes e responde de maneira adequada com base na intenção detectada.

## Funcionalidades

- **Detecção de Intenções**: Identifica intenções do usuário com base em mensagens de texto.
- **Respostas Personalizadas**: Gera respostas específicas para cada intenção e substitui `<HUMAN>` pelo nome do usuário.
- **Histórico de Intenções**: Armazena e permite a consulta das últimas três intenções detectadas.
- **Treinamento do Modelo**: Treina o modelo BERT se um modelo salvo não estiver disponível.

## Tecnologias Utilizadas

- Python
- PyTorch
- Transformers (Hugging Face)
- Scikit-Learn
- Pandas

## Estrutura do Projeto

- `chatbot.py`: Código principal do chatbot.
- `intent.json`: Conjunto de dados de intenções com textos de exemplo e respostas.
- `requirements.txt`: Arquivo de dependências para instalar bibliotecas necessárias.

## Pré-requisitos

- Python 3.7 ou superior
- PyTorch
- Transformers
- Scikit-Learn
- Pandas

Instale todas as dependências executando:

```bash
pip install -r requirements.txt
```

## Uso

1. **Treinamento e Execução**:
   - O chatbot treina automaticamente o modelo se um arquivo de modelo salvo (`intent_model.pth`) não for encontrado. Caso o modelo já exista, ele é carregado diretamente.
   
2. **Executando o Chatbot**:
   - Execute o chatbot com o comando:
   
     ```bash
     python chatbot.py
     ```

   - O chatbot solicitará o nome do usuário e estará pronto para receber mensagens de texto.

3. **Comandos Especiais**:
   - **Consultar Intenções**: Digite "consultar intenções" para ver as últimas três intenções detectadas.
   - **Encerrar Conversa**: Digite "sair" para finalizar a sessão.

## Exemplo de Uso

```
Olá! Eu sou o seu assistente virtual.
Por favor, qual é o seu nome? Nathan
Prazer em conhecê-lo, Nathan!
Nathan: How are you?
Nathan, sua intenção é: CourtesyGreeting
Resposta: Hello, I am great, how are you? Please tell me your GeniSys user
Nathan: What is my name?
Nathan, sua intenção é: CurrentHumanQuery
Resposta: They call you Nathan, what can I do for you?
```

## Estrutura do `intent.json`

O arquivo `intent.json` é um arquivo JSON coletados em https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset que contém as intenções, exemplos de texto para cada intenção e as respostas correspondentes. Estrutura básica:

```json
{
  "intents": [
    {
      "intent": "Greeting",
      "text": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello, <HUMAN>!", "Hi <HUMAN>, how can I help you today?"]
    },
    {
      "intent": "CourtesyGreeting",
      "text": ["How are you?", "How's it going?"],
      "responses": ["I'm doing well, <HUMAN>. How about you?"]
    }
  ]
}
```

## Estrutura do Código

### Arquivo `chatbot.py`

- **Importações**: Carrega as bibliotecas necessárias.
- **Configurações**: Define as configurações do modelo e hiperparâmetros.
- **Carregamento e Preparação de Dados**: Extrai dados de intenções, converte texto em tokens e organiza os dados para treinamento.
- **Definição do Modelo**: Modelo BERT com camada linear para classificação.
- **Funções**:
  - `train()`: Treina o modelo com o conjunto de dados fornecido.
  - `predict_intent()`: Realiza a previsão da intenção com base na entrada do usuário.
  - `personalize_response()`: Substitui `<HUMAN>` pelo nome do usuário na resposta.
  - `main()`: Funcionalidade principal do chatbot, gerencia a interação com o usuário.
