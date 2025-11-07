import os
import time
from typing import List, Dict, Any
import tempfile

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader

from agent import rag_agent
from vectorstore import add_document_to_vectorstore

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for the LangGraph-powered RAG agent with pinecone and Groq."
    # version="1.0.0"
)

# IN-MEMORY SESSION MANAGER
memory = MemorySaver()

# ------- PyDantic Models for API ---------
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type:  str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    enabled_web_search: bool = True

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int

# Document upload endpoint
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile=File(...)):
    """
        Uploads a PDF document, extract text, and adds it to the RAG knowledge base.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = "Only pdf files are supported"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file_content = await file.read()
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    print(f"Recieved PDF for upload : {file.filename}. Saved temporarly to {temp_file_path}")

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        total_chunks_added = 0
        if documents:
            # full_text_content = "\n\n".join(doc.page_content for doc in documents)
            full_text_content = ""
            for doc in documents:
                full_text_content += "\n\n" + doc.page_content
            add_document_to_vectorstore(full_text_content)
            total_chunks_added=len(documents)
        return DocumentUploadResponse(
            message=f"PDF '{file.filename}' successfully uploaded and indexed.",
            filename=file.filename,
            processed_chunks=total_chunks_added
        )

    except Exception as e:
        print(f"Error processing PDF document : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Failed to process PDF : {e}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temprorily file :{temp_file_path}")

@app.post("/chat/", response_model=AgentResponse)
async def chat_with_agent(request: QueryRequest):
    trace_event_for_front :List[TraceEvent] = []
    try:
        # passing web info the config for the agent to access
        config = {
            "configurable": {
                "thread_id": request.session_id,
                "web_search_enabled": request.enabled_web_search
            }
        }

        inputs = {"messages": [HumanMessage(content=request.query)]}
        final_message = " "
        # for server side debugging
        print(f"Starting Agent: stream for session {request.session_id}---")
        print(f"web search enabled: {request.enabled_web_search}")

        for i,s in enumerate(rag_agent.stream(inputs, config=config)):
            current_node_name=None
            node_output_state = None

            if '__end__' in s:
                current_node_name = '__end__'
                node_output_state = s['__end__']
            else:
                current_node_name = list(s.keys())[0]
                node_output_state = s[current_node_name]
            
            event_description = f"Execution node : {current_node_name}"
            event_details = {}
            event_type = "generic_node_execution"

            if current_node_name == "router":
                route_decision = node_output_state.get('route')
                #check for overridden route if web search was disabled
                initial_decision = node_output_state.get('initial_router_decision', route_decision)
                override_reason = node_output_state.get('router_override_reason', None)

                if override_reason:
                    event_description = f"Router initially decided: '{initial_decision}'. Overridden to: '{route_decision}' because {override_reason}"
                    event_details = {"initial_decision": initial_decision, "final_decision": route_decision, "override_reason": override_reason}
                else:
                    event_description = f"Router decided: '{route_decision}'"
                    event_details = {"decision": route_decision, "reason": "Based on initial query analysis."}
                event_type = "router_decision"
            elif current_node_name == "rag_lookup":
                rag_content_summary = node_output_state.get("rag", "")[:200] + "..."
                rag_sufficient = node_output_state.get("route") == "answer"

                if rag_sufficient:
                    event_description = f"RAG Lookup performed. Content found and deemed sufficient. Processing to answer."

                    event_details = {"retrived_content_summary": rag_content_summary, "sufficiency_verdict": "Sufficient"}
                else:
                    event_description = f"RAG Lookup performed. Content NOT sufficient. Diverting to web search."
                    event_details = {"retrienved_content_summary": rag_content_summary, "sufficiency_verdict": "Not Sufficient"}
                event_type = "rag_action"
            elif current_node_name == "web_search":
                web_conent_summary = node_output_state.get("web","")[:200]+"..."
                event_description = f"Web Search performed. Results retieved. Proceeding to answer."
                event_details = {"retrieved_content_sumary": web_conent_summary}
                event_type ="web_action"
            elif current_node_name == "answer":
                event_description = "Generating final answer using gathered context."
                event_type = "answer_generation"
            elif current_node_name == "__end__":
                event_description = "Agent process completed."
                event_type = "process_end"

            trace_event_for_front.append(
                TraceEvent(
                    step=i+1,
                    node_name=current_node_name,
                    description=event_description,
                    details=event_details,
                    event_type = event_type
                )
            )

        # Get the final state from the last yield item in the stream
        final_actual_state_dict = None
        if s:
            if '__end__' in s:
                final_actual_state_dict = s['__end__']
            else:
                if list(s.keys()):
                    final_actual_state_dict = s[list(s.keys())[0]]
        if final_actual_state_dict and "messages" in final_actual_state_dict:
            for msg in reversed(final_actual_state_dict["messages"]):
                if isinstance(msg, AIMessage):
                    final_message = msg.content
                    break
        if not final_message:
            print("Agent finished, but no final AIMessage found in the final state after stream completion.")
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Agent did not return a valid response (final AI message not found).")
        print(f"--- Agent Stream Ended. final Resposne: {final_message[:200]}... ----")

        return AgentResponse(response=final_message, trace_events=trace_event_for_front)

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_details = f"Error during agent invocation: {e}"
        print(error_details)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Internal Server Error : {e}")
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}