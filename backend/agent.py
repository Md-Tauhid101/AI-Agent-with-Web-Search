# Import dependencies
import os
from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from vectorstore import get_retriever
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
tavily = TavilySearch(max_results = 3, topic = "general")

@tool
def web_search_tool(query: str)->str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"

# Tools
@tool
def rag_search_tool(query: str) -> str:
    """Top-k Chunk from knowledge Base (empty strin  if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query, k=3)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"


# Define Pydantic Schema
class RouteDecision(BaseModel):
    route: Literal['rag', 'web', 'answer', 'end']
    reply: str | None = Field(None, description="Filled only when route == 'end' ")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, Fasle otherwise.")

# LLM instance with structured output
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

route_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Define STATE
class AgentState(TypedDict, total = False):
    messages: List[BaseMessage]
    route: Literal['rag', 'web', 'answer', 'end']
    rag: str
    web: str
    web_search_enabled: bool

# Create Nodes

# 1st Node
def router_node(state: AgentState) -> AgentState:
    print("Entering router_node")

    # Extract latest HumanMessage content
    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    web_search_enabled = state.get("web_search_enabled", True)
    print(f"Router received web search info: {web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool. "
        "Prioritize using the internal knowledge base (RAG) for factual information likely present in uploaded documents."
    )

    if web_search_enabled:
        system_prompt += (
            " You CAN use web search for very current, real-time, or broad knowledge not in the KB."
            " Choose one route: 'rag', 'web', 'answer', or 'end'."
        )
    else:
        system_prompt += (
            " Web search is DISABLED. You MUST NOT choose 'web'. Use 'rag' or 'answer' as appropriate."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    # Ensure override var exists and consistent spelling
    router_override_reason = None

    try:
        result: RouteDecision = route_llm.invoke(messages)
    except Exception as e:
        print(f"Router LLM error: {e}")
        # fallback: route to rag
        return {**state, "route": "rag", "web_search_enabled": web_search_enabled, "messages": state.get("messages", [])}

    initial_router_decision = result.route

    # Override if web disabled
    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_override_reason = "Web search disabled by user; redirected to rag"
        print(f"Router override: {initial_router_decision} -> {result.route}")

    out = {
        "messages": state.get("messages", []),
        "route": result.route,
        "web_search_enabled": web_search_enabled
    }

    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state.get("messages", []) + [AIMessage(content=result.reply or "Hello!")]

    print("Exiting router_node")
    return out



# 2nd Node : RAG LOOKUP
def rag_node(state: AgentState) -> AgentState:
    print("Entering rag_node")

    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    web_search_enabled = state.get("web_search_enabled", True)
    print(f"RAG Query: {query}")

    chunks = ""
    try:
        chunks = rag_search_tool.invoke(query)
    except Exception as e:
        print(f"RAG tool error: {e}")
        chunks = f"RAG_ERROR::{e}"

    if isinstance(chunks, str) and chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error: {chunks}")
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if chunks:
        print(f"Retrieved RAG chunks: {chunks[:500]}...")
    else:
        print("No RAG chunk retrieved")

    judge_message = [
        {
            "role": "system",
            "content": (
                "You are a judge that determines if retrieved information is sufficient and relevant to answer the question. "
                "Return whether it is sufficient or not using the schema: {\"sufficient\": true/false}."
            )
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nRetrieved info: {chunks}"
        }
    ]

    try:
        verdict: RagJudge = judge_llm.invoke(judge_message)
    except Exception as e:
        print(f"Judge LLM error: {e}")
        # conservative fallback: treat as not sufficient
        verdict = RagJudge(sufficient=False)

    print(f"RAG judge verdict: {verdict.sufficient}")

    # Use correct attribute name
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer"
        print(f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}")

    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "web_search_enabled": web_search_enabled
    }

# 3rd Node : Web Search
def web_node(state: AgentState) -> AgentState:
    print("Entering web_node")

    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    web_search_enabled = state.get("web_search_enabled", True)
    if not web_search_enabled:
        print("Web search node entered but web search is disabled")
        return {**state, "web": "Web search was disabled by user", "route": "answer"}

    print(f"web search query: {query}")
    try:
        snippets = web_search_tool.invoke(query)
    except Exception as e:
        print(f"Web search tool error: {e}")
        snippets = f"WEB_ERROR::{e}"

    if isinstance(snippets, str) and snippets.startswith("WEB_ERROR::"):
        print(f"web error: {snippets}. Proceeding to answer with limited info")
        return {**state, "web": "", "route": "answer"}

    print(f"web snippets retrieved: {snippets[:200]}")
    return {**state, "web": snippets, "route": "answer"}

# 4th Node: Final Answer
def answer_node(state: AgentState) -> AgentState:
    print("Entering answer_node")

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    context_parts = []
    if state.get("rag"):
        context_parts.append("Knowledge base information:\n" + state["rag"])
    if state.get("web"):
        # match the exact string used when web was disabled
        if state["web"] and not state["web"].lower().startswith("web search was disabled"):
            context_parts.append("Web search results:\n" + state["web"])

    context = "\n\n".join(context_parts).strip()
    if not context:
        context = "No external context was available for this query. Try to answer based on general knowledge."

    prompt = f"""Please answer the user's question using the provided context.
Question: {user_query}

Context: {context}
Provide a helpful, accurate, and concise response based on the available information.
"""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    try:
        ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(f"Answer LLM error: {e}")
        ans = "Sorry, I couldn't generate an answer at the moment."

    print(f"Final answer: {ans[:200]}...")
    return {**state, "messages": state.get("messages", []) + [AIMessage(content=ans)]}


# ---------- Routing helpers -------------------
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_)->Literal["answer"]:
    return "answer"

# Build Graph
def build_agent():
    """Builds and compiles the LangGraph agent."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("web_search", web_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")
    
    graph.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        }
    )
    
    graph.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer": "answer",
            "web": "web_search"
        }
    )
    
    graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)

    agent = graph.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()