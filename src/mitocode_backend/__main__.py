from backend.agent import mcp_manager

global_mcpmanager = mcp_manager.MCPManager()
global_mcpmanager.start()
