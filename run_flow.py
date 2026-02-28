import asyncio
import time

from app.agent.data_analysis import DataAnalysis
from app.agent.manus import Manus
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool import ToolCollection
from app.tool.web_search import WebSearch
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.ask_human import AskHuman
from app.tool.terminate import Terminate


class SearchAgent(ToolCallAgent):
    """A lightweight search agent that only uses web search tool"""
    name: str = "search"
    description: str = "A search agent specialized in web search tasks"
    max_steps: int = 3

    available_tools: ToolCollection = ToolCollection(
        WebSearch(),
        StrReplaceEditor(),
        AskHuman(),
        Terminate(),
    )


async def run_flow():
    # 创建 SearchAgent
    search_agent = SearchAgent()
    logger.info(f"[Flow] Created SearchAgent with tools: {[t.name for t in search_agent.available_tools.tools]}")

    # 创建 Manus Agent
    manus_agent = Manus()
    logger.info(f"[Flow] Created Manus with tools: {[t.name for t in manus_agent.available_tools.tools]}")

    agents = {
        "search": search_agent,
        "manus": manus_agent,
    }

    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()

    logger.info(f"[Flow] All agents: {list(agents.keys())}")

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )

        # 打印 planning_tool 的信息
        logger.info(f"[Flow] PlanningFlow initialized")
        logger.info(f"[Flow] executor_keys: {flow.executor_keys}")
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"[Flow] Request processed in {elapsed_time:.2f} seconds")
            logger.info(f"[Flow] Final result:\n{result}")

            # 打印执行历史
            if hasattr(flow, 'execution_history'):
                logger.info(f"[Flow] Execution history steps: {len(flow.execution_history)}")
                for i, entry in enumerate(flow.execution_history):
                    logger.info(f"[Flow] Step {i}: {entry.get('step_text', 'N/A')}")
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
