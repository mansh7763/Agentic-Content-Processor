from typing import Dict
from langchain.prompts import ChatPromptTemplate
from backend.llm.config import get_llm


def explain_code(code: str, language: str = None) -> Dict:
    llm = get_llm(temperature=0.2)
    
    language_hint = f"This appears to be {language} code." if language else "Detect the programming language."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert code reviewer and computer science educator.

Your task is to analyze code and provide:
1. EXPLANATION - A clear 2-3 sentence explanation of what the code does
2. BUGS - List any bugs, issues, or potential problems (or state "No obvious bugs detected")
3. COMPLEXITY - Time and space complexity in Big-O notation

Format your response EXACTLY like this:

EXPLANATION:
[Your clear explanation of what the code does]

BUGS:
- [Bug or issue 1]
- [Bug or issue 2]
OR
No obvious bugs detected

COMPLEXITY:
Time: O(n)
Space: O(1)

Be specific, educational, and helpful. Focus on helping the user understand the code."""),
        ("user", """{language_hint}

Code to analyze:
```
{code}
```

Please provide:
1. Explanation of what this code does
2. Any bugs or issues you detect
3. Time and space complexity analysis""")
    ])
    
    chain = prompt | llm
    
    try:
        print("Analyzing code with LLM...")
        response = chain.invoke({
            "code": code[:3000],  # Limit code length to avoid token limits
            "language_hint": language_hint
        })
        
        content = response.content
        result = parse_code_explanation(content)
        
        result["success"] = True
        result["language"] = language
        
        print("Code analysis complete!")
        return result
        
    except Exception as e:
        print(f"Error during code explanation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "explanation": "Error analyzing code",
            "bugs": ["Could not analyze for bugs"],
            "time_complexity": "Unknown",
            "space_complexity": "Unknown",
            "language": language
        }


def parse_code_explanation(content: str) -> Dict:
    explanation = ""
    bugs = []
    time_complexity = "Not specified"
    space_complexity = "Not specified"
    
    lines = content.strip().split('\n')
    current_section = None
    
    for line in lines:
        line_strip = line.strip()
        
        if line_strip.startswith("EXPLANATION:"):
            current_section = "explanation"
            exp_text = line_strip.replace("EXPLANATION:", "").strip()
            if exp_text:
                explanation = exp_text
                
        elif line_strip.startswith("BUGS:"):
            current_section = "bugs"
            bug_text = line_strip.replace("BUGS:", "").strip()
            if bug_text and bug_text.lower() != "bugs":
                bugs.append(bug_text)
                
        elif line_strip.startswith("COMPLEXITY:"):
            current_section = "complexity"
            
        elif line_strip.startswith("Time:"):
            time_complexity = line_strip.replace("Time:", "").strip()
            
        elif line_strip.startswith("Space:"):
            space_complexity = line_strip.replace("Space:", "").strip()
            
        elif line_strip and current_section == "explanation":
            explanation += " " + line_strip
            
        elif line_strip and current_section == "bugs":
            if line_strip.startswith("-") or line_strip.startswith("•") or line_strip.startswith("*"):
                bug = line_strip.lstrip("-•*").strip()
                if bug:
                    bugs.append(bug)
            elif "no" in line_strip.lower() and "bug" in line_strip.lower():
                bugs = ["No obvious bugs detected"]
            else:
                bugs.append(line_strip)
    
    if not explanation:
        explanation = "Code analysis completed"
    
    if not bugs:
        bugs = ["No specific bugs mentioned"]
    
    if len(bugs) > 1:
        bugs = [b for b in bugs if "no" not in b.lower() or "bug" not in b.lower()]
        if not bugs:
            bugs = ["No obvious bugs detected"]
    
    return {
        "explanation": explanation.strip(),
        "bugs": bugs,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity
    }