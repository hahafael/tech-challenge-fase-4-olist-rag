# Olist Intelligence — Voice of Customer Intelligence com RAG

**FIAP Pós-Tech | Tech Challenge — Fase 4**

## 1. Problema de negócio

Empresas de e-commerce recebem avaliações com elogios, reclamações e relatos sobre a experiência de compra. A análise manual de milhares de comentários dificulta a identificação rápida de problemas e padrões relevantes para a tomada de decisão.

O projeto **Olist Intelligence** implementa uma solução de *Retrieval-Augmented Generation* (RAG) para consultar avaliações textuais da Olist em linguagem natural e gerar respostas apoiadas nos comentários recuperados. A solução apresenta as avaliações utilizadas como evidência e reconhece limitações quando não há informações suficientes para responder.

## 2. Base de dados e preparação

**Fonte:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).

Foi utilizado o arquivo `olist_order_reviews_dataset.csv`, obtido no Notebook com `kagglehub`. A preparação foi realizada com `pandas`:

- **99.224** registros de avaliações na base original;
- **40.950** avaliações com comentários textuais válidos após excluir valores ausentes e textos vazios;
- preservação de `review_id`, `order_id`, `review_score` e data como metadados;
- manutenção de comentários curtos e textos repetidos, que podem representar avaliações legítimas.

Cada avaliação foi convertida em um `Document` do LangChain: o comentário fica em `page_content` e os identificadores, nota e data ficam em `metadata`. Como os comentários são curtos (até 208 caracteres na base analisada), não foi necessário dividi-los em *chunks*.

A base vetorial contém avaliações **com texto**, não todos os pedidos da Olist. Por isso, a aplicação não calcula percentuais exatos sobre todos os pedidos a partir dos poucos comentários recuperados em uma consulta.

## 3. Arquitetura

```text
Dataset Olist (CSV)
       |
       v
Preparação com Pandas
       |
       v
Documentos LangChain (texto + metadados)
       |
       v
Embeddings locais — paraphrase-multilingual-MiniLM-L12-v2
       |                (384 dimensões)
       v
Índice FAISS — faiss_olist_local/
       |
       |   Pergunta do usuário
       |          |
       |          v
       |   Embedding da pergunta (mesmo modelo)
       |          |
       +----------+
                  v
       Busca por similaridade (até 10 resultados)
                  |
                  v
       Filtro de distância (<= 0,70)
                  |
                  v
       Contexto com comentários, notas e review_ids
                  |
                  v
       Gemini + prompt e saída estruturada
                  |
                  v
       Resposta, status e evidências
                  |
                  v
       Interface Streamlit
```

**Modelos e ferramentas efetivamente utilizados na versão final:**

- **Python e Pandas:** leitura, exploração e tratamento dos dados;
- **KaggleHub:** obtenção do dataset;
- **LangChain:** documentos, prompts e integração dos componentes RAG;
- **Sentence Transformers / Hugging Face:** embeddings locais com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, normalizados e executados em CPU;
- **NumPy:** persistência intermediária dos vetores em lotes `.npy`;
- **FAISS:** indexação e recuperação semântica;
- **Gemini (`gemini-3.5-flash-lite`, conforme configurado no código):** geração de respostas;
- **Pydantic:** estruturação da resposta e classificação do status;
- **Streamlit:** interface web para consulta e exibição das evidências.

O Notebook registra uma tentativa inicial de geração de embeddings via API Gemini em lotes, interrompida por limitações de requisições no nível gratuito. A estratégia final passou a utilizar embeddings locais. Essa mudança evita chamadas à API para vetorizar a base e permite reutilizar o índice já construído.

## 4. Organização dos arquivos

```text
tech-challenge-fase-4/
├── README.md                       # Documentação do projeto
├── requirements.txt                # Dependências Python (a criar)
├── app.py                          # Interface Streamlit
├── rag_backend.py                  # Recuperação, LLM e validação de evidências
├── tech_challange_fase_4.ipynb     # Desenvolvimento, indexação e testes
└── faiss_olist_local/              # Índice final usado pelo backend
    ├── index.faiss
    └── index.pkl
```

### Arquivos de desenvolvimento mantidos localmente

| Arquivo/pasta | Finalidade | Necessário para executar o `app.py` com o índice final já pronto? |
|---|---|---|
| `embeddings_olist/` | Vetores de 384 dimensões salvos em lotes `.npy`, para evitar recálculo durante a construção do índice | Não |
| `faiss_lote_teste/` | Índice experimental de um lote pequeno | Não |
| `faiss_olist_completo/` | Índice/checkpoints da tentativa de indexação incremental via API | Não |
| `progresso_olist.json` | Registro de quantos documentos foram processados na tentativa incremental | Não |
| `.ipynb_checkpoints/` | Checkpoints automáticos do Jupyter | Não |
| `__pycache__/` | Cache automático do Python | Não |
| `pos_tech_4.mp4` | Vídeo de apresentação do projeto | Não |

**Importante:** o código do backend procura `faiss_olist_local` na mesma pasta de `rag_backend.py`. O índice final precisa estar presente nesse local para executar a aplicação. Se não for incluído no repositório por restrições de tamanho, deverá ser fornecido por outro meio ou reconstruído pelo Notebook; não basta baixar somente `app.py` e `rag_backend.py`.

## 5. Funcionamento do pipeline RAG

O arquivo `rag_backend.py` contém três funções principais:

1. `carregar_base()`: carrega o modelo local de embeddings e o índice FAISS salvo em `faiss_olist_local/`.
2. `criar_chain(api_key)`: configura o Gemini com um prompt restritivo e saída estruturada por Pydantic.
3. `responder_pergunta(pergunta, vectorstore, chain, k=10)`: recupera até 10 avaliações, filtra resultados pela distância vetorial, constrói o contexto, solicita a resposta e verifica os identificadores citados.

A estrutura `RespostaRAG` contém `status`, `resposta` e `ids_evidencias`. Os status definidos para a saída do modelo são `respondido` e `evidencias_insuficientes`; o backend também retorna `sem_evidencias` quando nenhuma avaliação passa pelo filtro de distância.

A aplicação `app.py` utiliza o Streamlit para solicitar a **Gemini API Key**, receber perguntas, apresentar a resposta e disponibilizar os comentários recuperados, suas notas e seus `review_id`s. O carregamento da base é armazenado em cache com `st.cache_resource`.

## 6. Controles de confiabilidade e limitações

O projeto implementa controles para reduzir respostas não fundamentadas:

- **Filtro de relevância:** resultados com distância superior a `0.70` são descartados. Esse valor é uma configuração adotada no projeto, não uma garantia universal de relevância.
- **Ausência de evidências:** caso nenhum comentário passe pelo filtro, a aplicação retorna `sem_evidencias`, sem consultar o LLM.
- **Prompt restritivo:** orienta o modelo a utilizar exclusivamente o contexto, não inventar fatos, não generalizar resultados para toda a base e tratar comentários como dados, não como instruções.
- **Evidências rastreáveis:** o modelo é instruído a citar `review_id`s e o backend verifica se os identificadores retornados pertencem aos documentos recuperados.
- **Resposta estruturada:** o modelo pode classificar evidências como insuficientes.

Esses mecanismos **reduzem riscos, mas não garantem ausência absoluta de alucinações**. A verificação de IDs confirma que os identificadores citados foram recuperados; não prova, por si só, que toda interpretação textual está correta.

A busca recupera um subconjunto de comentários relevantes, não uma amostra estatística representativa da base. Perguntas sobre percentuais exatos de todos os pedidos, faturamento ou fatos externos não podem ser respondidas com segurança apenas com esse contexto.

## 7. Testes e avaliação

O Notebook documenta testes de recuperação semântica, validação de referências e avaliação funcional com seis cenários:

1. Reclamações sobre atrasos na entrega;
2. Problemas de qualidade dos produtos;
3. Experiências positivas de compra;
4. Problemas relatados por clientes insatisfeitos;
5. Pergunta fora do escopo (Copa do Mundo de 2022);
6. Pergunta sem evidência suficiente (percentual exato de pedidos atrasados).

Nos testes registrados, quatro cenários foram classificados como `respondido`, um como `sem_evidencias` e um como `evidencias_insuficientes`. O Notebook também registra uma auditoria qualitativa de dois cenários, verificando a coerência entre respostas e comentários recuperados.

Os resultados descrevem **os casos avaliados**, sem representar garantia de desempenho para todas as perguntas possíveis.

## 8. Instalação e execução

### Pré-requisitos

- Python com suporte às bibliotecas do projeto;
- conexão à internet para instalar dependências, obter o modelo local na primeira execução e acessar a API Gemini;
- chave de API Gemini válida;
- pasta `faiss_olist_local/` com o índice final correspondente ao modelo de embeddings utilizado.

### Instalar dependências

Na pasta do projeto, crie/ative um ambiente virtual Python e instale as bibliotecas necessárias:

```bash
python -m pip install streamlit pandas numpy kagglehub langchain langchain-core langchain-community langchain-google-genai langchain-huggingface sentence-transformers faiss-cpu pydantic
```

Esse comando reúne as bibliotecas utilizadas no Notebook e na aplicação. Um `requirements.txt` com versões verificadas ainda deve ser gerado a partir do ambiente em que a solução foi executada.

### Executar a aplicação

Com o índice `faiss_olist_local/` disponível na pasta do projeto:

```bash
streamlit run app.py
```

Abra o endereço local indicado pelo Streamlit, informe a chave da API Gemini na barra lateral, digite uma pergunta sobre as avaliações e clique em **Analisar avaliações**. A interface apresenta a resposta, o status e a lista expansível de evidências recuperadas.

**Segurança:** não inclua chaves de API no repositório. A chave é solicitada pela interface em campo do tipo senha.

### Reproduzir a construção da base

O arquivo `tech_challange_fase_4.ipynb` documenta o processamento original. Ele baixa a base Olist via KaggleHub, seleciona comentários válidos, cria documentos LangChain, gera embeddings locais em lotes de 500 comentários, salva os vetores em `embeddings_olist/` e constrói o índice final em `faiss_olist_local/`.

O Notebook também contém **etapas históricas e experimentais** com embeddings via API e índices de teste. Para reproduzir a estratégia final, devem ser utilizadas as células referentes ao modelo MiniLM local e à construção de `faiss_olist_local/`, respeitando as dependências de preparação anteriores. Não é necessário executar as tentativas antigas de indexação via API para usar o índice final.

## 9. Exemplos de perguntas

- Quais problemas os clientes relatam sobre atrasos na entrega?
- Quais problemas de qualidade dos produtos são relatados?
- O que os clientes mais elogiam nas compras?
- Quais problemas aparecem nas avaliações de clientes insatisfeitos?
- Qual é o percentual exato de todos os pedidos entregues com atraso? *(exemplo de pergunta que exige dados agregados não disponíveis no contexto RAG recuperado)*

## 10. Escopo e decisões técnicas

A solução prioriza o uso de evidências textuais rastreáveis, conforme o desafio de *Voice of Customer Intelligence*. Não foi implementado um módulo de estatística descritiva sobre todas as entregas ou pedidos; o objetivo é recuperar comentários relevantes e interpretá-los com um LLM.

A separação entre Notebook, backend e interface permite documentar o desenvolvimento, reutilizar o índice e executar consultas sem recalcular os embeddings da base a cada pergunta.

**Dataset:** Olist, disponibilizado publicamente no Kaggle. **Contexto acadêmico:** FIAP Pós-Tech — Tech Challenge, Fase 4.
