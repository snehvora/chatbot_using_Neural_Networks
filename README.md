# Chatbot using Neural Networks 🤖

This project implements a simple **chatbot** trained using **Neural Networks** on conversational datasets. The chatbot learns dialogue patterns and generates responses to user queries.

## 🚀 Features
- Preprocesses conversational data (tokenization, cleaning, word-to-index mapping).
- Trains a neural network model (feedforward / LSTM / seq2seq depending on your implementation).
- Generates responses to user input after training.
- Includes dataset parsing (`movie_lines.txt`, `movie_conversations.txt`).

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy & Pandas**
- **NLTK / re**

## 📂 Project Structure
chatbot_using_Neural_Networks/<br>
│── chatbot.py # Main training and chatbot script.<br>
│── movie_conversations.txt # Dataset file (conversations).<br>
│── movie_lines.txt # Dataset file (dialogue lines).<br>
│── README.md # Project documentation.<br>

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/snehvora/chatbot_using_Neural_Networks.git
```
```bash
cd chatbot_using_Neural_Networks
```

Install dependencies:
```bash
pip3 install tensorflow numpy pandas nltk
```

## ▶️ Usage
Train the chatbot model:
```bash
python3 chatbot.py
```

## 📊 Dataset

After training, start chatting by typing into the console:
```bash
You: Hello!
Bot: Hi there, how are you doing?
```

This project uses the Cornell Movie Dialogs Corpus:<br>
- movie_lines.txt → dialogue lines<br>
- movie_conversations.txt → conversation mapping between lines<br>


## 🔮 Future Improvements
Add attention mechanism for better responses
- Save/load trained models
- Build a web or app interface for the chatbot

## 🤝 Contributing
Pull requests are welcome! If you find bugs or have feature suggestions, feel free to open an issue.
