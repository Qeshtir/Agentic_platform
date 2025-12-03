import argparse
from loguru import logger
from agent_platform.config import config_holder
from agent_platform.services.agent import SimpleAgent
from agent_platform.services.tools import get_weather, multiply, add, divide


if __name__ == "__main__":
    logger.info("Starting Agent service")

    parser = argparse.ArgumentParser(description="Agent service")
    parser.add_argument(
        "-c", dest="config_path", required=True, help="path to config file"
    )

    args = parser.parse_args()
    config_holder.load_config(args.config_path)

    logger.info("Starting server")
    config = config_holder.get_config()

    tools = [get_weather, multiply, add, divide]

    agent = SimpleAgent(config, tools)

    agent.build_agent()
    agent.print_agent()
    agent.invoke_agent()
    agent.invoke_agent(content="2 * 2")
