# Task 3: Multi-Tool Autonomous AI Agent

**Intern:** M. Hazaifa Tahir  
**Program:** Datheon Generative AI Internship  

## Overview
A console-based autonomous **ReAct AI Agent** built with **LangChain**, **LangGraph**, and **Google Gemini** capable of decomposing multi-step compound queries and dynamically invoking external tools while displaying its step-by-step reasoning trace (`Thought -> Action -> Observation -> Final Answer`).

## Equipped Tools
1. **`web_search` (`DuckDuckGoSearchRun`):** Fetches real-time information, exchange rates, and current events from the web.
2. **`math_calculator` (AST Safe Evaluator):** Safely parses and evaluates complex arithmetic expressions without unsafe `eval()`.
3. **`read_local_file`:** Reads local project files (such as `sample_notes.txt`) within the working directory.

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
