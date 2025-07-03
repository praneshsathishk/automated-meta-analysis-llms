from src.research_question_generator import generate_clarifying_questions, finalize_research_question

# Step 1: General user input
initial_input = "aspirin and heart attacks"

# Step 2: LLaMA asks clarifying questions
questions = generate_clarifying_questions(initial_input)
print("Clarifying Questions:\n", questions)

# Step 3: Simulated user clarification (you'd usually collect this interactively)
refined_input = "I'm interested in whether daily aspirin reduces the risk of first heart attack in adults over 65 who have never had cardiovascular disease."

# Step 4: LLaMA finalizes the research question
final_question = finalize_research_question(refined_input, prompt_type="few-shot")
print("\nFinal Research Question:\n", final_question)
