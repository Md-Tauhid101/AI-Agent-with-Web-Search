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
    suffient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, Fasle otherwise.")

# LLM instance with structured output
GROQ_API_KEY = os.environ("GROQ_API_KEY")

route_llm = ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)

# Define STATE
class AgentState(TypedDict, total = False):
    message: List[BaseMessage]
    route: Literal['rag', 'web', 'answer', 'end']
    rag: str
    web: str
    web_search_enabled: bool

# Create Nodes

# 1st Node
def router_node(state: AgentState) -> AgentState:
    print("Entering route node")
    # extract query
    query = ""
    if isinstance(msg, HumanMessage):
        for msg in reversed(state['message']):
            query = next(msg.content)
    else:
        query = ""
    web_search_enabled = state['web_search_enabled', True]
    print(f"Router received web search info :{web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "prioritize using the **internal knowledge base (RAG)** for factual information that is likely to be contained within pre-uploaded documents or for common, well-established facts."
    )

    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e-g-, today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e-g-, 'what is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e-g-, 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e-g., 'What is your name?'"
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e-g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions: "
        "\n- User: 'What are the treatment of diabetes?' -› Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'what is the capital of France?' › Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -› Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -› Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. if KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -› Route: 'end', reply='Hello! How can I assist you today?'"
    )

    # add message and query in list
    messages = [
        {"syatem", system_prompt},
        {"user", query}
    ]

    result: RouteDecision = route_llm.invoke(messages)
    initial_router_decision = result.route

    # Override the router decision to go for web search
    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_overrider_reason = "Web search disabled by user, redirected to rag"
        print(f"Router final decision: {result.route}, reply (if 'end'):{result.reply}")

    out = {
        "message": state['message'],
        "router": result.route,
        "web_search_enabled": web_search_enabled
    }

    if router_overrider_reason:
        out['initial_router_decision'] = initial_router_decision
        out['router_override_reason'] = router_overrider_reason

    if result.route == 'end':
        out['message'] = state['message'] + [AIMessage(content=result.reply or "Hello!")]

    print("Existing router_node")
    return out


# 2nd Node : RAG LOOKUP
def rag_node(state: AgentState) -> AgentState:
    print("Entering rag_node")
    query = ""
    if isinstance(msg, HumanMessage):
        for msg in reversed(state['message']):
            query = next(msg.content)
    else:
        query = ""
    web_search_enabled = state['web_search_enabled', True]
    print(f"Router received web search info :{web_search_enabled}")
    print("RAG Query: {query}")

    chunks = rag_search_tool.invoke(query)

    # logic to handle chunk
    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error :{chunks}, checking web search enabled status")
        # if rag fails, and web search is enabled
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if chunks:
        print(f"Retrieved RAG chunks: {chunks[:500]}...")
    else:
        print("No RAG chunk retrieved")

    judge_message = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            " If no relevant information was retrieved at all (e-g., 'No results found'), it is definitely NOT sufficient."
            "\n\n Respond ONLY with a JSON object: (\"sufficient\": true/false)"
            "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' "
            "-› \": true}"
            "\n Example 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -› (\"sufficient\": false} (Doesn't answer symptoms)"
            "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\". false}"
        )),
        {"user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?"}
    ]

    verdict: RagJudge=judge_llm.invoke(judge_message)
    print(f"RAG judge verdict : {verdict.sufficient}")
    print("Existing rag_node")

    # Decide next route based on sufficienct and web_search info
    if verdict.suffient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer"
        print("RAG not sufficient. Web search enabled : {web_search_enabled}.Next route:{next_route}")
    
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
    if isinstance(msg, HumanMessage):
        for msg in reversed(state['message']):
            query = next(msg.content)
    else:
        query = ""
    web_search_enabled = state['web_search_enabled', True]
    
    if not web_search_enabled:
        print("Web search node entered but web search is disabled")
        return {
            **state,
             "web": "Web search was disabled by user",
              "route": "answer" 
              }    

    print(f"web search query: {query}")
    snippts = web_search_tool.invoke(query)
    if snippts.startswith("WEB_ERROR::"):
        print(f"web error : {snippts}.Predicting to answer with limited info")
        return {
            **state,
            "web": "",
            "route": "answer"
        }
    print(f"web snippts retrived : {snippts[:200]}")
    print("Existing web_node")
    return {
        **state,
        "web": snippts,
        "route": "answer"
    }

# 4th Node: Final Answer
def answer_node(state: AgentState)->AgentState:
    print("Entring answer_node")
    user_query = ""
    if isinstance(msg, HumanMessage):
        for msg in reversed(state['message']):
            user_query = next(msg.content)
    else:
        user_query = ""
    
    context_part = []
    if state.get("rag"):
        context_part.append("Knoweledge base Information :\n"+state["rag"])
    elif state.get("web"):
        if state["web"] and not state["web"].startswith("web search was disabled"):
            context_part.append("web Search Results : \n"+state["web"])
    context = "\n\n".join(context_part)
    if not context.strip():
        context = "No external context was available for this query. Try to answer based on general knowledge."

    prompt = f"""
        Please answer the user's question using the provided context.
        If the context is empty or irrelevant, try to answer based on your general knowledge.

        Question: {user_query}

        Context: {context}
        Provide a helpful, accurate, and concise response based on the available information.
    """

    print(f"Prompt sent to answer_llm : {prompt[:500]}...")
    ans=answer_llm.invoke([HumanMessage(content = prompt)]).content
    print(f"Final answer : {ans[:200]}...")
    print("Existing answer_node")
    return{
        **state,
        "message": state["message"]+[AIMessage(content=ans)]
    }

# ---------- Routing helpers -------------------
def from_roter(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_)->Literal["answer"]:
    return "answer"

# Build Graph
def build_agent():
    """Build and compile the LangGraph """
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        from_roter,
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
            "web": "web_search",
            "answer": "answer"
        }
    )

    graph.add_conditional_edges(
        "web search",
        after_web,
        {
            "web": "web_search"
        }
    )

    graph.add_edge("answer", END)
    agent = graph.compile(checkpointer=MemorySaver)
    return agent

rag_agent = build_agent()