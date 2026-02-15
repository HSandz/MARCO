from marco.tools.base import Tool
from marco.tools.info_database import InfoDatabase
from marco.tools.interaction import InteractionRetriever
from marco.tools.candidate_retriever import CandidateRetriever
TOOL_MAP: dict[str, type] = {
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
    'retriever': CandidateRetriever,
}