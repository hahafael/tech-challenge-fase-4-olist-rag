
import streamlit as st

from rag_backend import (
    carregar_base,
    criar_chain,
    responder_pergunta
)

# Configuração da interface
st.set_page_config(
    page_title="Olist Intelligence",
    page_icon="🛍️",
    layout="wide"
)

st.title("Olist Intelligence")
st.caption("Voice of Customer Intelligence com RAG | FIAP")

st.write(
    "Consulte avaliações de clientes da Olist "
    "utilizando inteligência artificial."
)


# Carregar a base apenas uma vez
@st.cache_resource
def inicializar_base():
    return carregar_base()


# Barra lateral
with st.sidebar:
    st.header("Configurações")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    st.divider()
    st.subheader("Arquitetura")
    st.write("**Embeddings:** MiniLM multilíngue")
    st.write("**Busca:** FAISS")
    st.write("**LLM:** Gemini 3.5 Flash Lite")


# Carregar o índice
try:
    with st.spinner("Carregando base de conhecimento..."):
        vectorstore = inicializar_base()

    st.success(
        f"Base carregada: "
        f"{vectorstore.index.ntotal:,} avaliações."
    )

except Exception as erro:
    st.error(f"Erro ao carregar a base: {erro}")
    st.stop()


# Campo de pergunta
pergunta = st.text_area(
    "Faça uma pergunta sobre as avaliações:",
    placeholder="Ex.: Quais são as principais reclamações sobre entregas?"
)


if st.button("Analisar avaliações", type="primary"):

    if not api_key:
        st.warning("Informe sua Gemini API Key.")

    elif not pergunta.strip():
        st.warning("Digite uma pergunta.")

    else:
        try:
            with st.spinner("Analisando avaliações..."):
                chain = criar_chain(api_key)

                resultado = responder_pergunta(
                    pergunta.strip(),
                    vectorstore,
                    chain
                )

            st.divider()
            st.subheader("Resposta")

            st.write(resultado["resposta"])

            st.caption(
                f"Status: {resultado['status']} | "
                f"Evidências: {len(resultado['evidencias'])}"
            )

            with st.expander("Ver evidências recuperadas"):
                for doc in resultado["evidencias"]:
                    st.write(doc.page_content)
                    st.caption(
                        f"Nota: {doc.metadata['review_score']} | "
                        f"Review ID: {doc.metadata['review_id']}"
                    )
                    st.divider()

        except Exception as erro:
            st.error(f"Erro na consulta: {erro}")
