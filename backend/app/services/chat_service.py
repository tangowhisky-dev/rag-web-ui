import json
import base64
import logging
import re  # used by _IDENTITY_PATTERNS
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase
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
    logger.info("=" * 70)
    logger.info("[CHAT] chat_id=%s | kb_ids=%s | query=%r", chat_id, knowledge_base_ids, query)

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
            logger.info("[CHAT] identity shortcut — skipping RAG")
            yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = _IDENTITY_RESPONSE
            db.commit()
            return

        # Get knowledge bases
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )

        if not knowledge_bases:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return

        # Build chat history from previous messages (exclude the current/last message)
        logger.info("[CHAT] total messages in payload=%d | prior (history) messages=%d",
                    len(messages["messages"]), len(messages["messages"]) - 1)
        chat_history = []
        prior_messages = messages["messages"][:-1]
        for message in prior_messages:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                content = message["content"]
                if "__LLM_RESPONSE__" in content:
                    content = content.split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=content))

        # Step 1: Condense question with chat history into a standalone question
        logger.info("[STEP 1] condense | chat_history_turns=%d", len(chat_history))
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
                    "reformulate it if needed and otherwise return it as is."
                    "Your only task is to rewrite the user's question as a fully self-contained question "
                    "that can be understood without the chat history. "
                    "Output ONLY the rewritten question — no explanations, no answers, no extra text. "
                    "If the question is already self-contained, output it unchanged. "
                    "Never answer the question.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            logger.info("[STEP 1] sending condense request to model=%s", settings.OPENAI_MODEL)
            condense_chain = contextualize_q_prompt | llm | StrOutputParser()
            raw_rewrite = (await condense_chain.ainvoke(
                {"input": query, "chat_history": chat_history}
            )).strip()
            # Strip reasoning blocks emitted by thinking models (e.g. <think>...</think>)
            had_think = bool(re.search(r"<think>", raw_rewrite))
            standalone_question = re.sub(r"<think>.*?</think>", "", raw_rewrite, flags=re.DOTALL).strip()
            if not standalone_question:
                standalone_question = query
            logger.info("[STEP 1] raw_rewrite=%r | had_think_block=%s | standalone_question=%r",
                        raw_rewrite[:300], had_think, standalone_question)

        logger.info("[STEP 1] standalone_question=%r", standalone_question)

        # Step 2: Retrieve relevant documents via hybrid search (dense + sparse + BM25 + RRF)
        logger.info("[STEP 2] starting hybrid_search | standalone_question=%r", standalone_question)
        docs = await hybrid_search(
            query=standalone_question,
            kb_ids=knowledge_base_ids,
            db=db,
        )

        logger.info("[STEP 2] hybrid_search returned %d docs", len(docs))
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:120].replace("\n", " ")
            logger.info("  chunk[%d] meta=%s | text=%r", i, doc.metadata, snippet)

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
            "You are a professional AI-based Knowledge Assistant that answers questions using the provided context documents.\n\n"
            "## Formatting\n"
            "- Use **bold** for key terms, concepts, and important phrases.\n"
            "- Use *italics* for definitions, technical terms, or emphasis.\n"
            "- Use numbered lists (1. 2. 3.) for sequential steps or ordered items.\n"
            "- Use bullet points (- or *) for non-ordered lists, features, or comparisons.\n"
            "- Use headings (##, ###) only for longer multi-section answers.\n"
            "- Keep paragraphs short and well-separated for readability.\n\n"
            "## Citations\n"
            "You will be given context documents numbered sequentially starting from 1.\n"
            "You MUST cite sources using EXACTLY this format: [citation:x] — for example: 'The sky is blue [citation:1].'\n"
            "Do NOT use any other citation format such as [1], (1), Context [1], or footnotes.\n"
            "If a sentence draws from multiple contexts, list all applicable citations: [citation:1][citation:2].\n\n"
            "## General\n"
            "- Your answer must be correct, accurate, and written in a professional, unbiased tone.\n"
            "- Limit your response to 2048 tokens.\n"
            "- Do not include information unrelated to the question, and do not repeat yourself.\n"
            "- If the provided context does not contain sufficient information, say so briefly and professionally.\n"
            "- Write in the same language as the question (except for code, citations, and proper nouns).\n"
            "- Do not blindly repeat the contexts verbatim; synthesise and explain.\n\n"
            "Context:\n{context}"
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

        logger.info("[STEP 4] QA request | model=%s | messages=%d | context_chunks=%d",
                    settings.OPENAI_MODEL, len(openai_messages), len(docs))
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

        logger.info("[STEP 4] QA streaming complete | response_length=%d chars", len(full_response))
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