# Finix

https://youtu.be/5flql9_1OD4

Finix é uma aplicação mobile educativa que ensina, na prática, as diferenças e consequências entre investir e apostar. Voltada para crianças e jovens, combina minijogos rápidos, mecânica de pontos/estrelas e recompensas in‑game para promover alfabetização financeira por meio de decisões simuladas.

---

## Sumário
- Visão geral  
- Recursos principais  
- Protótipo de reconhecimento emocional  
- Requisitos  
- Instalação rápida  
- Uso  
- Arquitetura e pipeline técnico  
- Coleta de dados e treino do modelo  
- Integração com backend e frontend (futuro)  
- Privacidade e ética  
- Estrutura do repositório  
- Roadmap  
- Contato

---

## Visão geral
Finix permite ao usuário ganhar pontos/estrelas em minijogos. Com esses recursos, o usuário pode optar por:
- apostar (alto risco, retorno imprevisível) ou  
- investir (baixo/médio risco, retorno mais previsível).

Resultados afetam recompensas estéticas (skins, itens) dentro dos jogos. O objetivo é ensinar, com experiências práticas e feedback, as diferenças entre apostar e investir.

---

## Recursos principais
- Minijogos rápidos que geram pontos/estrelas.  
- Sistema de decisão: gastar pontos em apostas ou investimentos simulados.  
- Recompensas in‑game desbloqueáveis (skins, itens).  
- Detecção opcional de emoções negativas (raiva, tristeza) via câmera frontal quando o usuário aceita.  
- Logs e CSVs de predições para análise e treinamento futuro.

---

## Protótipo de reconhecimento emocional
O protótipo detecta emoções negativas (foco em raiva) usando MediaPipe e um pipeline em Python. A demonstração inclui:
- processamento de vídeo pré‑gravado (ex.: cena de Star Wars: Revenge of the Sith)  
- captura por webcam frontal (com consentimento no app final).

Funcionalidades do protótipo:
- extração de Face Mesh (landmarks faciais) com MediaPipe;  
- features geométricas e temporais;  
- inferência por heurística ou classificador treinado (RandomForest);  
- suavização temporal e gravação de resultados em CSV.

---

## Requisitos
- Python 3.8+  
- Pacotes Python:
  - mediapipe
  - opencv-python
  - numpy
  - scikit-learn
  - joblib
  - pandas

---

## Arquitetura e pipeline técnico

Fluxo:

- Captura do frame (vídeo ou webcam).

- Face Mesh (MediaPipe) → landmarks.

- Normalização de pose (interpupilar) e centralização.

- Extração de features instantâneas (ex.: mouth_ratio, mouth_width_rel, brow_avg, nose_mouth_rel, brow_angle).

- Cálculo de features temporais (velocidade média, aceleração média, desvios padrão das landmarks chave).

- Concatenação do vetor de features → inferência (heurística ou RandomForest).

- Suavização temporal das predições e gravação de logs.

---

## Privacidade e ética
- Consentimento explícito para uso de câmera frontal (opt‑in).

- Controle do usuário para desativar coleta e excluir dados.

- Minimização: armazenar vetores anonimizados sempre que possível.

- Transparência: explicar quando e por que a câmera é usada.

- Avaliar e mitigar vieses; evitar decisões automatizadas de alto impacto sem supervisão humana.

---

## Roadmap
 - **Curto prazo:** integrar registro de eventos emocionais ao backend e criar endpoints de alerta.

 - **Médio prazo:** implementar recomendações automatizadas com o módulo educativo e testes A/B de UX.

 - **Longo prazo:** treinar e integrar modelos multimodais, otimizar para mobile e realizar testes de campo.

---

## Integrantes:
- João Pedro Borsato da Cruz: rm550294
- Maria Fernanda Vieira de Camargo: rm97956
- Pedro Lucas de Andrade Nunes: rm550366
- Sofia Amorim Coutinho: rm552534
