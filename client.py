import asyncio
import json
import ollama
from fastmcp import Client

MCP_URL = "https://f1pulse-mcp.onrender.com/mcp"
MODEL = "llama3.1:8b"

def to_ollama_tools(mcp_tools):
    #take the tools from mcp server and return them in ollama format
    
    return[
        {
            "type" : "function",
            "function" : {
                "name" : t.name,
                "description" : t.description,
                "parameters" : t.inputSchema,
            },
        }
        for t in mcp_tools
    ]

async def run_turn(client, ollama_tools, messages):
    #this is to keep looping until the model gives a plain answer (5 rounds max)
    
    for _ in range(5):
        response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
        msg = response.message
        messages.append(msg) #it remembers what the model said
        
        
        #if theres no tool call = model answered. return it to user.
        if not msg.tool_calls:
            return msg.content
        
        #otherwise run the tool it asked for
        for call in msg.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            print(f" -> {name}({args})") #shows the call
            
            #if the tool fails, tell the model instead of crashing
            try:
                result = await client.call_tool(name, args)
                data = result.data if result.data is not None else result.content
                content = json.dumps(data, default=str)
            except Exception as e:
                content = f"Tool error: {e}"
                print(f"      ERROR: {e}")
            
            #sends the result back to the model
            messages.append({"role": "tool", "tool_name": name, "content": content})
    
    return "Too many tool calls, stopping."
        
        
        
async def main():
    async with Client(MCP_URL, auth="oauth") as client:
        mcp_tools = await client.list_tools()
        ollama_tools = to_ollama_tools(mcp_tools)  # this is to convert once
        print(f"Connected - {len(mcp_tools)} tools. Ask an F1 question (quit to exit. \n)")
        
        #system prompt = tells the model to answer in words, not just dump the whole json file
        messages = [{"role": "system", "content":
            "You are an F1 assistant with tools. Call a tool when needed,"
            "then answer in plain words. Never output raw JSON or repeat the tool call"
            }]
        
        #chat loop
        while True:
            user = input("you> ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            
            if not user:
                continue
            
            messages.append({"role": "user", "content": user}) #adds question
            answer = await run_turn(client, ollama_tools, messages) #runs loop
            print(f"\nbot> {answer} \n")

if __name__ == "__main__":
    asyncio.run(main())