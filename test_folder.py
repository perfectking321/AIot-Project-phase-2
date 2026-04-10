import sys
import logging
from agent.dispatcher import CommandDispatcher

logging.basicConfig(level=logging.DEBUG)

command = "create a folder named testfolder on my desktop"
print(f"Testing command: {command}")

dispatcher = CommandDispatcher(None)
result = dispatcher.dispatch(command)
print(f"Result: {result}")
