
import asyncio
import json
import sys
from pathlib import Path
import os

# Set environment variables to ensure settings are loaded correctly
os.environ["PRISM_VAULT_PATH"] = "/home/kingy/Foundation/NimbusVault"
os.environ["PRISM_DATA_DIR"] = "/home/kingy/Foundation/PrismRag/data"

# Add project root to path to allow imports
sys.path.insert(0, '/home/kingy/Foundation/PrismRag')

from prism_rag.mcp_server.server import mcp, _ensure_federated

async def main():
    # Ensure the graph is loaded, same as in the server
    _ensure_federated()

    if len(sys.argv) < 2:
        print("Usage: python run_mcp_tool.py <tool_name> [args_json]")
        return

    tool_name = sys.argv[1]
    arguments = {}
    if len(sys.argv) > 2:
        try:
            arguments = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            # Try to handle positional args if passed as multiple strings (simple case)
            # But better to just use JSON for complexity
            print(f"Error: Second argument must be a JSON string of arguments. Got: {sys.argv[2]}")
            sys.exit(1)

    try:
        # FastMCP.call_tool is async
        result = await mcp.call_tool(tool_name, arguments)
        
        # result is likely a list of content objects (TextContent, etc.)
        # We want the text content.
        if isinstance(result, list):
            for item in result:
                if hasattr(item, 'text'):
                    print(item.text)
                else:
                    print(item)
        else:
            print(result)

    except Exception as e:
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
