# Import functions from the ollama library for AI chat and web tools
from ollama import chat, web_fetch, web_search

def execute_prompt(prompt):
  # Create a dictionary to map tool names to their corresponding functions
  # This allows the AI to call these tools by name during the conversation
  available_tools = {'web_search': web_search, 'web_fetch': web_fetch}

  # Initialize the conversation with a user question
  # The messages list maintains the entire conversation history for context
  messages = [{'role': 'user', 'content': prompt}]

  # Main conversation loop - continues until AI doesn't need to use tools
  while True:
    # Send the conversation history to the AI model
    response = chat(
      model='gpt-oss:20b',           # Use the GPT-OSS 20B model
      messages=messages,             # Pass the entire conversation history
      tools=[web_search, web_fetch], # Make web tools available to the AI
      think=True                     # Enable thinking mode to show AI's reasoning
      )
    
    # Display the AI's internal thinking process if available
    if response.message.thinking:
      print('Thinking: ', response.message.thinking)
    
    # Display the AI's main response content
    if response.message.content:
      print('Content: ', response.message.content)
    
    # Add the AI's response to conversation history to maintain context
    messages.append(response.message)
    
    # Check if the AI wants to use any tools (web search or web fetch)
    if response.message.tool_calls:
      print('Tool calls: ', response.message.tool_calls)
      
      # Process each tool call the AI requested
      for tool_call in response.message.tool_calls:
        # Look up the requested function in our available tools
        function_to_call = available_tools.get(tool_call.function.name)
        
        if function_to_call:
          # Extract the arguments the AI wants to pass to the tool
          args = tool_call.function.arguments
          
          # Execute the tool with the provided arguments
          result = function_to_call(**args)
          
          # Display a truncated version of the result for readability
          print('Result: ', str(result)[:200]+'...')
          
          # Add the full result to conversation (truncated to prevent context overflow)
          # Result is truncated for limited context lengths
          messages.append({'role': 'tool', 'content': str(result)[:2000 * 4], 'tool_name': tool_call.function.name})
        else:
          # Handle case where AI requests a tool that doesn't exist
          messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
    else:
      # If no tool calls were made, the AI has finished and we can exit the loop
      break

# Main execution block - only runs when script is executed directly (not imported)
if __name__ == "__main__":
    import sys
    
    # Check if user provided a prompt as command line argument
    if len(sys.argv) < 2:
        # Show usage instructions if no prompt was provided
        # print("Usage: python oweb.py <prompt>")
        prompt = "What is stellar blade?"
    else:
        # Extract the prompt from command line arguments
        prompt = sys.argv[1]
        
    # Execute the AI conversation with the provided prompt
    execute_prompt(prompt)