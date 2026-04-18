import json
import base64
import re  # used by _IDENTITY_PATTERNS
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from app.services.retrieval import hybrid_search

_IDENTITY_PATTERNS = re.compile(
    r"^\s*(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"what\s+is\s+your\s+name|what('s| is)\s+your\s+purpose|what\s+can\s+you\s+do)\s*\??\s*$",
    re.IGNORECASE,
)

_IDENTITY_RESPONSE = (
    "I'm professional AI based Knowledge Assistant that answers questions using "
    "the documents and knowledge bases you've uploaded. "
    "Ask me anything about your content and I'll retrieve the most relevant information "
    "and give you a clear, cited answer."
)


def _is_identity_question(query: str) -> bool:
    return bool(_IDENTITY_PATTERNS.match(query.strip()))


async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()

        # Short-circuit identity questions — no RAG needed
        if _is_identity_question(query):
            yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = _IDENTITY_RESPONSE
            db.commit()
            return

        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings and vector store client
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            model=settings.OPENAI_EMBEDDINGS_MODEL,
            check_embedding_ctx_length=False,
        )
        chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_DB_HOST,
            port=settings.CHROMA_DB_PORT,
        )

        # Create a vector store for each knowledge base
        vector_stores = []
        for kb in knowledge_bases:
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            if documents:
                vector_store = Chroma(
                    client=chroma_client,
                    collection_name=f"kb_{kb.id}",
                    embedding_function=embeddings,
                )
                vector_stores.append(vector_store)
        
        if not vector_stores:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return

        # Build chat history from previous messages
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                content = message["content"]
                if "__LLM_RESPONSE__" in content:
                    content = content.split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=content))

        # Step 1: Condense question with chat history into a standalone question
        standalone_question = query
        if chat_history:
            llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, just "
                    "reformulate it if needed and otherwise return it as is.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            condense_chain = contextualize_q_prompt | llm | StrOutputParser()
            standalone_question = await condense_chain.ainvoke(
                {"input": query, "chat_history": chat_history}
            )

        # Step 2: Retrieve relevant documents via hybrid search (dense + BM25 + RRF)
        docs = await hybrid_search(
            query=standalone_question,
            kb_ids=knowledge_base_ids,
            db=db,
            vector_stores=vector_stores,
        )

        # Step 3: Emit context chunk as base64 before streaming the answer
        serializable_context = [
            {
                "page_content": doc.page_content.replace('"', '\\"'),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        base64_context = base64.b64encode(
            json.dumps({
                "context": serializable_context,
                "rewritten_query": standalone_question,
            }).encode()
        ).decode()
        separator = "__LLM_RESPONSE__"
        yield f'0:"{base64_context}{separator}"\n'
        full_response = base64_context + separator

        # Step 4: Stream the QA answer via LCEL (LangChain 1.x)
        formatted_context = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
        )
        qa_system_prompt = (
            "You are a professional AI based Knowledge Assistant that answers questions using the provided context documents. "
            "You will be given a set of related contexts to the question, numbered sequentially starting from 1. "
            "Each context has an implicit reference number based on its position in the list (first context is 1, second is 2, etc.). "
            "You MUST cite sources using EXACTLY this format: [citation:x] — for example: 'The sky is blue [citation:1].' "
            "Do NOT use any other citation format such as [1], (1), Context [1], or footnotes. "
            "Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. "
            "Please limit to 2048 tokens. Do not give any information that is not related to the question, and do not repeat. "
            "If the provided context does not contain sufficient information to answer the question, say so briefly and professionally. "
            "If a sentence draws from multiple contexts, list all applicable citations: [citation:1][citation:2]. "
            "Other than code and specific names and citations, your answer must be written in the same language as the question. "
            "Be concise.\n\nContext: {context}\n\n"
            "Remember: cite using [citation:x] only. Do not blindly repeat the contexts verbatim."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_messages = qa_prompt.format_prompt(
            input=query,
            chat_history=chat_history,
            context=formatted_context,
        ).to_messages()

        openai_messages = []
        for message in qa_messages:
            role = "user"
            if isinstance(message, AIMessage):
                role = "assistant"
            elif message.type == "system":
                role = "system"

            openai_messages.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )

        stream = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=openai_messages,
            temperature=0,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            chunk_text = delta.content or ""

            if not chunk_text:
                continue

            full_response += chunk_text
            yield f'0:{json.dumps(chunk_text)}\n'

        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        yield '3:{text}\n'.format(text=error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()