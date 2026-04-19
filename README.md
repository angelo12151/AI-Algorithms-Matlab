# AI Algorithms From Scratch (MATLAB) 🧠

Implementazione manuale di architetture di Deep Learning e modelli lineari senza l'ausilio di toolbox predefiniti. Il progetto si focalizza sulla comprensione algoritmica profonda e sull'ottimizzazione numerica dei parametri.

## 🚀 Key Features

### 1. Multi-Layer Perceptron (MLP) Engine (`anetwork.m`)
Classe flessibile per la creazione di reti neurali dense:
- **Backpropagation**: Calcolo manuale dei gradienti per pesi e bias.
- **SGD**: Stochastic Gradient Descent con supporto per mini-batch.
- **Learning Rate Scheduler**: Adattamento dinamico della velocità di apprendimento ($\eta$) per ottimizzare la convergenza.
- **Versatilità**: Supporto per compiti di classificazione (Sigmoide/Softmax) e regressione (Lineare).

### 2. Rosenblatt Perceptron (`perceptron.m`)
- Modello ad oggetti per la classificazione binaria lineare.
- Gestione delle epoche di addestramento e verifica della separabilità dei dati.

## 📈 Demos & Use Cases

Il repository include script dimostrativi per i tre pilastri del Machine Learning:

1. **Separazione Non Lineare (`esempio1.m`)**: Classificazione di cluster complessi tramite MLP e visualizzazione dei Decision Boundaries.
2. **Multi-class Classification (`esempio2.m`)**: Classificazione chimica dei vini (3 classi) con valutazione tramite Confusion Matrix.
3. **Regressione (`esempio3.m`)**: Approssimazione di funzioni non lineari tramite output activation lineare e calcolo del MSE.

## 🛠️ Tech Stack
- **Linguaggio**: MATLAB.
- **Competenze**: Algebra lineare applicata, ottimizzazione stocastica, calcolo differenziale.

---
*Progetto sviluppato per coniugare il rigore matematico con l'efficienza algoritmica.*
