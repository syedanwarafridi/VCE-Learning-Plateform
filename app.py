import streamlit as st
import psycopg2
import json
import requests
from typing import Dict, Any, List
import time
from datetime import datetime

# ==================== DATABASE FUNCTIONS ====================
def get_db_connection():
    """Establish connection to PostgreSQL database"""
    return psycopg2.connect(
        host="localhost",
        database="vce_learning_platform",
        user="postgres",
        password="postgres1234",
        port="5432"
    )

def get_questions_list(limit=20):
    """Get a list of questions for selection"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT q.question_id, q.question_number, q.question_text, 
               e.year, e.subject, e.exam_name, q.difficulty_level
        FROM questions q
        JOIN exams e ON q.exam_id = e.exam_id
        ORDER BY q.question_id
        LIMIT %s;
    """, (limit,))
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    questions = []
    for row in rows:
        questions.append({
            "question_id": row[0],
            "question_number": row[1],
            "preview_text": row[2][:150] + "..." if len(row[2]) > 150 else row[2],
            "year": row[3],
            "subject": row[4],
            "exam_name": row[5],
            "difficulty": row[6]
        })
    
    return questions

def get_question_by_id(question_id):
    """Get complete question data by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT  
            q.question_id, q.question_number, q.section, q.unit, q.aos, q.subtopic,
            q.skill_type, q.difficulty_level, q.question_text, q.answer_text, 
            q.detailed_answer, q.page_number,

            e.exam_id, e.year, e.subject, e.unit AS exam_unit, e.exam_name,
            e.pdf_url, e.source, e.scraped_at,

            ab.aos_name, ab.percentage,

            sp.subpart_id, sp.subpart_letter, sp.subpart_text, 
            sp.subpart_answer, sp.subpart_detailed_answer

        FROM questions q
        JOIN exams e ON q.exam_id = e.exam_id
        LEFT JOIN aos_breakdown ab ON ab.exam_id = e.exam_id
        LEFT JOIN question_subparts sp ON sp.question_id = q.question_id
        WHERE q.question_id = %s
        ORDER BY sp.subpart_letter;
    """, (question_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return None

    result = {
        "question_id": rows[0][0],
        "question_number": rows[0][1],
        "section": rows[0][2],
        "unit": rows[0][3],
        "aos": rows[0][4],
        "subtopic": rows[0][5],
        "skill_type": rows[0][6],
        "difficulty_level": rows[0][7],
        "question_text": rows[0][8],
        "answer_text": rows[0][9],
        "detailed_answer": rows[0][10],
        "page_number": rows[0][11],
        "exam": {
            "exam_id": rows[0][12],
            "year": rows[0][13],
            "subject": rows[0][14],
            "unit": rows[0][15],
            "exam_name": rows[0][16],
            "pdf_url": rows[0][17],
            "source": rows[0][18],
            "scraped_at": rows[0][19]
        },
        "aos_breakdown": [],
        "subparts": []
    }

    for r in rows:
        if r[20]: 
            result["aos_breakdown"].append({"aos_name": r[20], "percentage": r[21]})
        if r[22]: 
            result["subparts"].append({
                "subpart_id": r[22],
                "subpart_letter": r[23],
                "subpart_text": r[24],
                "subpart_answer": r[25],
                "subpart_detailed_answer": r[26]
            })

    return result

# ==================== GRANITE API FUNCTIONS ====================
def query_granite(user_prompt, system_prompt="You are a math reasoning assistant.", context="", 
                  api_url="https://nab6wk9x0oev1u-8888.proxy.runpod.net/api/granite/generate", timeout=300):
    """Send query to Granite model API"""
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context": context
    }

    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("output", "No output found in response.")
        else:
            return f"❌ Non-200 response: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Model may still be loading."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to the server. Check the proxy URL."
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# ==================== PROMPT TEMPLATES ====================
MARKING_PROMPTS = {
    "evaluate_solution": """You are an expert mathematics examiner for VCE (Victorian Certificate of Education) exams.

EVALUATION TASK:
Compare the student's solution against the correct answer and provide:
1. ✅ CORRECT or ❌ INCORRECT verdict
2. Score out of 10
3. Step-by-step feedback
4. Common mistakes to avoid
5. Suggested improvements

QUESTION:
{question_text}

CORRECT ANSWER (from exam):
{detailed_answer}

STUDENT'S SOLUTION:
{student_solution}

ADDITIONAL CONTEXT:
- Subject: {subject}
- Year: {year}
- Exam: {exam_name}
- Area of Study: {aos}
- Difficulty: {difficulty}
- Skill Type: {skill_type}

Provide your evaluation in this exact format:
VERDICT: [✅ CORRECT or ❌ INCORRECT]
SCORE: [X/10]
FEEDBACK: [Detailed feedback here...]
MISTAKES: [Common mistakes section...]
IMPROVEMENTS: [Suggested improvements...]""",

    "explain_correct_answer": """You are a mathematics tutor explaining a VCE exam question solution.

QUESTION:
{question_text}

CORRECT SOLUTION:
{detailed_answer}

STUDENT'S ATTEMPT:
{student_solution}

Explain the correct solution clearly, highlighting:
1. Key concepts tested
2. Step-by-step reasoning
3. How it differs from the student's approach (if incorrect)
4. Tips for similar problems

Make your explanation engaging and educational.""",

    "rubric_evaluation": """As a VCE mathematics assessor, evaluate this solution against the official marking rubric.

QUESTION DETAILS:
{question_text}
Difficulty: {difficulty}
Skill Type: {skill_type}
Area of Study: {aos}

RUBRIC CRITERIA:
1. Conceptual Understanding (0-3 points)
2. Procedural Accuracy (0-3 points)
3. Problem Solving Strategy (0-2 points)
4. Communication of Reasoning (0-2 points)

STUDENT'S SOLUTION:
{student_solution}

CORRECT ANSWER:
{detailed_answer}

Provide rubric scores and brief justification for each criterion."""
}

TUTOR_PROMPTS = {
    "general_tutor": """You are a highly experienced VCE Mathematics tutor with expertise across:
- Mathematical Methods (CAS and non-CAS)
- Specialist Mathematics
- Further Mathematics
- Foundation Mathematics

You excel at:
1. Breaking down complex problems into manageable steps
2. Using VCE-specific terminology and notation
3. Relating concepts to VCAA study design
4. Providing multiple solution approaches when appropriate
5. Highlighting common pitfalls and exam techniques

Always structure your explanations with:
1. **Understanding the Problem** - What's being asked, key terms
2. **Relevant Theory** - VCAA Study Design references
3. **Step-by-Step Solution** - Clear, logical progression
4. **Check & Verify** - How to verify the answer
5. **Key Takeaways** - Summary of learning points

Use VCE-appropriate mathematical notation and terminology.""",

    "methods_tutor": """You are a VCE Mathematical Methods specialist tutor.

Key VCAA Study Design Areas you excel in:
- Functions and graphs (polynomial, exponential, logarithmic, circular)
- Calculus (differentiation, integration, applications)
- Probability and statistics (discrete/continuous random variables, distributions)

Teaching Approach:
1. Start with what the student knows
2. Connect to prior learning
3. Use CAS/non-CAS appropriate methods
4. Emphasize practical applications
5. Include exam-style practice tips

Always reference:
- Appropriate technology use (CAS calculators)
- VCAA examination report insights
- Common student errors from past exams
- Efficient solving techniques for exam conditions""",

    "specialist_tutor": """You are a VCE Specialist Mathematics expert tutor.

VCAA Specialist Mathematics Focus Areas:
- Vector calculus and kinematics
- Complex numbers and polar forms
- Differential equations and modelling
- Mechanics and proof techniques

Specialist Mathematics Pedagogy:
1. Emphasize rigorous mathematical reasoning
2. Connect theoretical concepts to physical applications
3. Demonstrate elegant solution methods
4. Highlight connections between different areas of mathematics
5. Prepare students for proof-based questions

Include in explanations:
- Formal mathematical notation
- Proof techniques when applicable
- Real-world applications (physics, engineering)
- Extension material for high-achieving students""",

    "step_by_step": """You are explaining a mathematical concept to a VCE student who wants to understand it thoroughly.

Follow this exact 5-step framework:

**Step 1: Problem Analysis**
- Restate the problem in your own words
- Identify key mathematical concepts involved
- Note any constraints or special conditions

**Step 2: Theory Review**
- Briefly recall relevant formulas, theorems, or definitions
- Reference VCAA Study Design points if applicable
- Connect to previously learned concepts

**Step 3: Detailed Solution Walkthrough**
- Break into logical substeps
- Show all working clearly
- Explain the "why" behind each step
- Include diagrams if helpful (describe verbally)
- Use proper mathematical notation

**Step 4: Verification & Alternative Approaches**
- Check the solution makes sense
- Suggest alternative methods if applicable
- Point out common errors to avoid
- Discuss how to verify the answer

**Step 5: Learning Extension**
- Summarize key techniques learned
- Suggest similar practice problems
- Connect to exam question types
- Provide study tips for this topic

Always use student-friendly language while maintaining mathematical rigor."""
}

# ==================== STREAMLIT APPLICATION ====================
def main():
    st.set_page_config(
        page_title="VCE Mathematics Learning Platform",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* ================= GENERAL ================= */
    body, .stApp {
        background-color: #000000;
        color: #e5e7eb;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #111827;
        color: #e5e7eb;
        border: 1px solid #374151;
    }

    /* ================= MARKING SYSTEM ================= */
    .correct-answer {
        background-color: #020617;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        margin: 10px 0;
    }

    .incorrect-answer {
        background-color: #020617;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        margin: 10px 0;
    }

    .question-card {
        background-color: #020617;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #334155;
    }

    .feedback-box {
        background-color: #020617;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #334155;
    }

    /* ================= CHAT ================= */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }

    .chat-message.user {
        background-color: #020617;
        border-left: 5px solid #3b82f6;
    }

    .chat-message.assistant {
        background-color: #020617;
        border-left: 5px solid #22c55e;
    }

    /* ================= TUTOR SECTIONS ================= */
    .step-box {
        background-color: #020617;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    .vce-highlight {
        background-color: #020617;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #22c55e;
        margin: 10px 0;
    }

    .exam-tip {
        background-color: #020617;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
    }

    /* ================= TABS ================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #020617;
        color: #e5e7eb;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #334155;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #334155;
    }

    .sidebar-title {
        color: #93c5fd;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* ================= EXPANDERS ================= */
    .streamlit-expanderHeader {
        background-color: #020617;
        color: #e5e7eb;
        border: 1px solid #334155;
    }

    .streamlit-expanderContent {
        background-color: #020617;
        border: 1px solid #334155;
    }

    /* ================= INPUTS ================= */
    textarea, input, select {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border: 1px solid #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
    # Initialize session state for all pages
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎓 VCE Learning Platform</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "Select Mode",
            ["🧠 AI Tutor Chat", "📝 Marking System", "📚 Question Bank"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Platform Stats
        st.markdown("### 📊 Platform Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", "500+")
        with col2:
            st.metric("Exams", "50+")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if page == "🧠 AI Tutor Chat":
            if st.button("🔄 Clear Chat History"):
                st.session_state.tutor_messages = []
                st.rerun()
        
        elif page == "📝 Marking System":
            if st.button("🎲 Load Random Question"):
                load_random_question()
        
        elif page == "📚 Question Bank":
            if st.button("🔄 Refresh Questions"):
                st.session_state.questions_list = get_questions_list(limit=20)
                st.rerun()
        
        st.markdown("---")
        
        # About Section
        with st.expander("ℹ️ About this Platform"):
            st.info("""
            **VCE Mathematics Learning Platform**
            
            Features:
            - 🧠 AI Tutor: Step-by-step explanations
            - 📝 Marking System: Rubric-based evaluation
            - 📚 Question Bank: Real VCE exam questions
            - 🎯 Curriculum-aligned: VCAA Study Design
            
            Powered by Granite AI Model
            """)
    
    # Main Content Area
    if page == "🧠 AI Tutor Chat":
        show_tutor_chat()
    elif page == "📝 Marking System":
        show_marking_system()
    else:  # Question Bank
        show_question_bank()

def init_session_state():
    """Initialize all session state variables"""
    # Marking System
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None
    if 'student_solution' not in st.session_state:
        st.session_state.student_solution = ""
    if 'feedback_result' not in st.session_state:
        st.session_state.feedback_result = None
    if 'questions_list' not in st.session_state:
        st.session_state.questions_list = []
    
    # Tutor Chat
    if 'tutor_messages' not in st.session_state:
        st.session_state.tutor_messages = []
    if 'tutor_mode' not in st.session_state:
        st.session_state.tutor_mode = "general_tutor"
    
    # Question Bank
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0

def load_random_question():
    """Load a random question for marking system"""
    questions = get_questions_list(limit=1)
    if questions:
        st.session_state.selected_question = get_question_by_id(questions[0]['question_id'])
        st.session_state.student_solution = ""
        st.session_state.feedback_result = None

# ==================== TUTOR CHAT PAGE ====================
def show_tutor_chat():
    """Teaching/Tutoring System Interface"""
    st.title("🧠 VCE Mathematics AI Tutor")
    st.markdown("### Your Personal Tutor for Mathematical Methods & Specialist Mathematics")
    
    # Tutor configuration sidebar (within main area for better layout)
    with st.sidebar:
        st.header("🎯 Tutor Configuration")
        
        tutor_mode = st.selectbox(
            "Select Tutor Specialization",
            options=[
                ("General Mathematics Tutor", "general_tutor"),
                ("Mathematical Methods Specialist", "methods_tutor"),
                ("Specialist Mathematics Expert", "specialist_tutor"),
                ("Step-by-Step Framework", "step_by_step")
            ],
            format_func=lambda x: x[0],
            index=0
        )
        st.session_state.tutor_mode = tutor_mode[1]
        
        st.markdown("---")
        st.header("📘 VCE Curriculum Focus")
        
        curriculum_area = st.multiselect(
            "Select Areas of Study",
            [
                "Functions and Graphs",
                "Algebra", 
                "Calculus",
                "Probability and Statistics",
                "Vectors and Matrices",
                "Complex Numbers",
                "Differential Equations",
                "Proofs and Logic"
            ],
            default=["Functions and Graphs", "Calculus"]
        )
        
        year_level = st.select_slider(
            "Year Level",
            options=["Unit 1/2", "Unit 3/4"]
        )
        
        include_exam_tips = st.checkbox("Include Exam Tips", value=True)
        include_cas = st.checkbox("Include CAS Instructions", value=True)
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.tutor_messages:
            # Welcome message
            st.markdown("""
            <div class="chat-message assistant">
                <strong>👋 Welcome to your VCE Mathematics AI Tutor!</strong><br><br>
                I'm here to help you with:
                - Understanding mathematical concepts
                - Solving practice problems
                - Preparing for exams
                - Clarifying difficult topics<br><br>
                <em>Ask me anything about VCE Mathematics!</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.tutor_messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Format the assistant's response
                    formatted_response = format_tutor_response(message["content"])
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <strong>AI Tutor:</strong><br>
                        {formatted_response}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Quick example questions
    st.markdown("### 💡 Try These Questions:")
    example_cols = st.columns(4)
    
    examples = [
        ("Find domain of √(x²-4x+3)", "general_tutor"),
        ("Explain chain rule with examples", "methods_tutor"),
        ("Solve complex number: (2+3i)/(1-i)", "specialist_tutor"),
        ("Derivative of inverse trig functions", "step_by_step")
    ]
    
    for idx, (example, mode) in enumerate(examples):
        with example_cols[idx]:
            if st.button(example[:25] + "...", key=f"example_{idx}", use_container_width=True):
                st.session_state.tutor_mode = mode
                process_tutor_input(example, curriculum_area, year_level, include_exam_tips, include_cas)
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Ask your mathematics question:",
            placeholder="E.g., 'Explain how to find the domain of √(x² - 4x + 3)' or 'Help me understand integration by parts'",
            height=100,
            key="tutor_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📤 Send", use_container_width=True, type="primary"):
            if user_input.strip():
                process_tutor_input(user_input, curriculum_area, year_level, include_exam_tips, include_cas)
            else:
                st.warning("Please enter a question.")

def process_tutor_input(user_input, curriculum_area, year_level, include_exam_tips, include_cas):
    """Process user input and get tutor response"""
    # Add user message to chat
    st.session_state.tutor_messages.append({
        "role": "user", 
        "content": user_input
    })
    
    # Prepare context based on sidebar selections
    context_info = f"""
    Additional Context for Tutor:
    - Curriculum Areas: {', '.join(curriculum_area) if curriculum_area else 'General'}
    - Year Level: {year_level}
    - Include Exam Tips: {include_exam_tips}
    - Include CAS Instructions: {include_cas}
    
    Student Question: {user_input}
    """
    
    # Get system prompt based on selected mode
    system_prompt = TUTOR_PROMPTS.get(
        st.session_state.tutor_mode, 
        TUTOR_PROMPTS["general_tutor"]
    )
    
    # Add curriculum-specific guidance
    if curriculum_area:
        system_prompt += f"\n\nFocus specifically on: {', '.join(curriculum_area)}"
    
    if year_level == "Unit 3/4":
        system_prompt += "\n\nFocus on Unit 3/4 content with exam preparation emphasis."
    
    if include_exam_tips:
        system_prompt += "\n\nInclude practical exam tips and common VCAA marking scheme insights."
    
    if include_cas:
        system_prompt += "\n\nInclude CAS calculator instructions where applicable."
    
    # Show loading and get response
    with st.spinner("🤔 AI Tutor is thinking..."):
        response = query_granite(
            user_prompt=user_input,
            system_prompt=system_prompt,
            context=context_info
        )
    
    # Add assistant response to chat
    st.session_state.tutor_messages.append({
        "role": "assistant", 
        "content": response
    })
    
    st.rerun()

def format_tutor_response(response_text):
    """Format the tutor's response with enhanced visual elements"""
    # Simple formatting - you can enhance this based on your needs
    return response_text.replace("**Step", "<br>**Step").replace("**Tip:", "<br>**Tip:")

# ==================== MARKING SYSTEM PAGE ====================
def show_marking_system():
    """Marking System Interface"""
    st.title("📝 VCE Mathematics Marking System")
    st.markdown("### AI-Powered Practice & Evaluation Platform")
    
    # Load questions if not loaded
    if not st.session_state.questions_list:
        with st.spinner("Loading questions from database..."):
            st.session_state.questions_list = get_questions_list(limit=20)
    
    # Sidebar for question selection
    with st.sidebar:
        st.header("📚 Question Selection")
        
        # Search and filter
        search_term = st.text_input("🔍 Search questions", placeholder="Type keywords...")
        
        # Filter by difficulty
        difficulties = ["All"] + sorted(list(set([q['difficulty'] for q in st.session_state.questions_list if q['difficulty']])))
        selected_difficulty = st.selectbox("Filter by difficulty", difficulties)
        
        # Filter by subject
        subjects = ["All"] + sorted(list(set([q['subject'] for q in st.session_state.questions_list if q['subject']])))
        selected_subject = st.selectbox("Filter by subject", subjects)
        
        # Filtered questions
        filtered_questions = st.session_state.questions_list
        
        if search_term:
            filtered_questions = [q for q in filtered_questions 
                                if search_term.lower() in q['preview_text'].lower() 
                                or search_term.lower() in str(q['question_number']).lower()]
        
        if selected_difficulty != "All":
            filtered_questions = [q for q in filtered_questions if q['difficulty'] == selected_difficulty]
        
        if selected_subject != "All":
            filtered_questions = [q for q in filtered_questions if q['subject'] == selected_subject]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question display area
        if st.session_state.selected_question:
            question_data = st.session_state.selected_question
            
            # Display question header
            st.subheader(f"Question {question_data['question_number']}")
            
            # Metadata
            with st.expander("📋 Question Details", expanded=True):
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.markdown(f"**Year:** {question_data['exam']['year']}")
                    st.markdown(f"**Subject:** {question_data['exam']['subject']}")
                with col_meta2:
                    st.markdown(f"**Difficulty:** {question_data['difficulty_level']}")
                    st.markdown(f"**Skill Type:** {question_data['skill_type']}")
                with col_meta3:
                    st.markdown(f"**AOS:** {question_data['aos']}")
                    st.markdown(f"**Section:** {question_data['section']}")
            
            # Display question text
            st.markdown('<div class="question-card">', unsafe_allow_html=True)
            st.markdown(f"**{question_data['question_number']}.** {question_data['question_text']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display subparts if they exist
            if question_data['subparts']:
                st.subheader("📝 Subparts")
                for subpart in question_data['subparts']:
                    with st.expander(f"Subpart {subpart['subpart_letter']}"):
                        st.write(subpart['subpart_text'])
                        st.caption(f"**Answer:** {subpart['subpart_answer']}")
            
            # Student solution input
            st.subheader("✍️ Your Solution")
            student_solution = st.text_area(
                "Write your solution here:",
                value=st.session_state.student_solution,
                height=200,
                placeholder="Enter your step-by-step solution here...\nShow all working and reasoning.",
                key="solution_input",
                label_visibility="collapsed"
            )
            
            # Action buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                submit_button = st.button("✅ Submit for Marking", use_container_width=True, type="primary")
            with col_btn2:
                show_answer = st.button("📖 Show Answer", use_container_width=True)
            with col_btn3:
                clear_button = st.button("🔄 Clear", use_container_width=True)
            
            if clear_button:
                st.session_state.student_solution = ""
                st.session_state.feedback_result = None
                st.rerun()
            
            if show_answer:
                with st.expander("📘 Correct Solution", expanded=True):
                    st.markdown(question_data['detailed_answer'])
            
            # Process submission
            if submit_button and student_solution:
                with st.spinner("🔍 Evaluating your solution with AI..."):
                    # Prepare prompt for Granite
                    prompt = MARKING_PROMPTS["evaluate_solution"].format(
                        question_text=question_data['question_text'],
                        detailed_answer=question_data['detailed_answer'],
                        student_solution=student_solution,
                        subject=question_data['exam']['subject'],
                        year=question_data['exam']['year'],
                        exam_name=question_data['exam']['exam_name'],
                        aos=question_data['aos'],
                        difficulty=question_data['difficulty_level'],
                        skill_type=question_data['skill_type']
                    )
                    
                    # Get AI feedback
                    feedback = query_granite(
                        user_prompt=prompt,
                        system_prompt="You are an expert VCE mathematics examiner providing detailed feedback."
                    )
                    
                    # Store feedback in session state
                    st.session_state.feedback_result = feedback
                    st.session_state.student_solution = student_solution
                    
                    st.rerun()
            
            # Display feedback if available
            if st.session_state.feedback_result:
                display_feedback(question_data)
        
        else:
            # No question selected
            st.info("👈 Select a question from the sidebar to get started!")
            st.markdown("""
            ### How to use the Marking System:
            1. Select a question from the sidebar
            2. Read the question carefully
            3. Write your solution in the text box
            4. Click "Submit for Marking"
            5. Review AI feedback and improve
            
            ### Features:
            - ✅ Rubric-based scoring
            - 📝 Detailed step-by-step feedback
            - ⚠️ Common mistake identification
            - 🚀 Personalized improvement suggestions
            """)
    
    with col2:
        # Question list in sidebar
        st.subheader("Available Questions")
        
        if not filtered_questions:
            st.warning("No questions match your filters.")
        else:
            for idx, q in enumerate(filtered_questions[:15]):  # Show first 15
                is_selected = (st.session_state.selected_question and 
                             st.session_state.selected_question['question_id'] == q['question_id'])
                
                container = st.container()
                
                with container:
                    if is_selected:
                        st.markdown("""
                        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; 
                                    border-left: 4px solid #2196f3; margin: 5px 0;">
                        """, unsafe_allow_html=True)
                    
                    col_sel1, col_sel2 = st.columns([4, 1])
                    with col_sel1:
                        st.markdown(f"**Q{q['question_number']}** ({q['year']})")
                        st.caption(f"{q['subject']} - {q['difficulty']}")
                        st.caption(q['preview_text'][:80] + "...")
                    with col_sel2:
                        if st.button("Select", key=f"select_{idx}", type="primary" if is_selected else "secondary"):
                            with st.spinner(f"Loading question {q['question_id']}..."):
                                st.session_state.selected_question = get_question_by_id(q['question_id'])
                                st.session_state.student_solution = ""
                                st.session_state.feedback_result = None
                                st.rerun()
                    
                    if is_selected:
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")

def display_feedback(question_data):
    """Display the feedback from AI evaluation"""
    st.markdown("---")
    st.subheader("🎯 AI Feedback & Evaluation")
    
    feedback_text = st.session_state.feedback_result
    
    # Parse feedback sections
    verdict = ""
    score = ""
    feedback_content = ""
    mistakes = ""
    improvements = ""
    
    lines = feedback_text.split('\n')
    current_section = ""
    
    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
            current_section = "verdict"
        elif line.startswith("SCORE:"):
            score = line.replace("SCORE:", "").strip()
            current_section = "score"
        elif line.startswith("FEEDBACK:"):
            feedback_content = line.replace("FEEDBACK:", "").strip()
            current_section = "feedback"
        elif line.startswith("MISTAKES:"):
            mistakes = line.replace("MISTAKES:", "").strip()
            current_section = "mistakes"
        elif line.startswith("IMPROVEMENTS:"):
            improvements = line.replace("IMPROVEMENTS:", "").strip()
            current_section = "improvements"
        elif current_section == "feedback" and line.strip() and not line.startswith("MISTAKES:"):
            feedback_content += "\n" + line.strip()
        elif current_section == "mistakes" and line.strip() and not line.startswith("IMPROVEMENTS:"):
            mistakes += "\n" + line.strip()
        elif current_section == "improvements" and line.strip():
            improvements += "\n" + line.strip()
    
    # Display verdict with appropriate styling
    if "✅ CORRECT" in verdict:
        st.markdown(f'<div class="correct-answer"><h3>🎉 {verdict}</h3></div>', unsafe_allow_html=True)
        st.balloons()
    elif "❌ INCORRECT" in verdict:
        st.markdown(f'<div class="incorrect-answer"><h3>📝 {verdict}</h3></div>', unsafe_allow_html=True)
    elif verdict:
        st.markdown(f'**{verdict}**')
    
    # Display score
    if score:
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score2:
            st.metric("Your Score", score)
    
    # Display feedback sections in tabs
    tab1, tab2, tab3 = st.tabs(["📝 Detailed Feedback", "⚠️ Common Mistakes", "🚀 Improvements"])
    
    with tab1:
        if feedback_content:
            st.markdown(feedback_content)
        else:
            st.info("No detailed feedback available.")
    
    with tab2:
        if mistakes:
            st.markdown(mistakes)
        else:
            st.info("No specific mistakes identified.")
    
    with tab3:
        if improvements:
            st.markdown(improvements)
        else:
            st.info("No improvement suggestions available.")
    
    # Additional analysis options
    st.markdown("---")
    st.subheader("🔍 Additional Analysis")
    
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        if st.button("🤔 Explain Correct Solution", use_container_width=True):
            with st.spinner("Generating explanation..."):
                explain_prompt = MARKING_PROMPTS["explain_correct_answer"].format(
                    question_text=question_data['question_text'],
                    detailed_answer=question_data['detailed_answer'],
                    student_solution=st.session_state.student_solution
                )
                explanation = query_granite(
                    user_prompt=explain_prompt,
                    system_prompt="You are a patient mathematics tutor explaining concepts clearly."
                )
                with st.expander("📘 Step-by-Step Explanation", expanded=True):
                    st.markdown(explanation)
    
    with col_anal2:
        if st.button("📊 Rubric Breakdown", use_container_width=True):
            with st.spinner("Analyzing against rubric..."):
                rubric_prompt = MARKING_PROMPTS["rubric_evaluation"].format(
                    question_text=question_data['question_text'],
                    detailed_answer=question_data['detailed_answer'],
                    student_solution=st.session_state.student_solution,
                    difficulty=question_data['difficulty_level'],
                    skill_type=question_data['skill_type'],
                    aos=question_data['aos']
                )
                rubric_analysis = query_granite(
                    user_prompt=rubric_prompt,
                    system_prompt="You are a VCE mathematics assessor applying marking rubrics."
                )
                with st.expander("📋 Rubric Evaluation", expanded=True):
                    st.markdown(rubric_analysis)

# ==================== QUESTION BANK PAGE ====================
def show_question_bank():
    """Question Bank Interface"""
    st.title("📚 VCE Question Bank")
    st.markdown("### Browse and Search VCE Mathematics Questions")
    
    # Load questions if not loaded
    if not st.session_state.questions_list:
        with st.spinner("Loading questions from database..."):
            st.session_state.questions_list = get_questions_list(limit=50)
    
    # Search and filter controls
    col_search1, col_search2, col_search3, col_search4 = st.columns(4)
    
    with col_search1:
        search_term = st.text_input("🔍 Search", placeholder="Keyword...")
    
    with col_search2:
        year_filter = st.multiselect(
            "Year",
            options=sorted(list(set([str(q['year']) for q in st.session_state.questions_list]))),
            default=[]
        )
    
    with col_search3:
        subject_filter = st.multiselect(
            "Subject",
            options=sorted(list(set([q['subject'] for q in st.session_state.questions_list if q['subject']]))),
            default=[]
        )
    
    with col_search4:
        difficulty_filter = st.multiselect(
            "Difficulty",
            options=sorted(list(set([q['difficulty'] for q in st.session_state.questions_list if q['difficulty']]))),
            default=[]
        )
    
    # Filter questions
    filtered_questions = st.session_state.questions_list
    
    if search_term:
        filtered_questions = [q for q in filtered_questions 
                            if search_term.lower() in q['preview_text'].lower() 
                            or search_term.lower() in str(q['question_number']).lower()
                            or search_term.lower() in str(q['year']).lower()]
    
    if year_filter:
        filtered_questions = [q for q in filtered_questions if str(q['year']) in year_filter]
    
    if subject_filter:
        filtered_questions = [q for q in filtered_questions if q['subject'] in subject_filter]
    
    if difficulty_filter:
        filtered_questions = [q for q in filtered_questions if q['difficulty'] in difficulty_filter]
    
    # Display question count
    st.markdown(f"**Showing {len(filtered_questions)} of {len(st.session_state.questions_list)} questions**")
    
    # Display questions in a grid
    if filtered_questions:
        # Pagination
        items_per_page = 10
        total_pages = max(1, (len(filtered_questions) + items_per_page - 1) // items_per_page)
        
        # Page selector
        if total_pages > 1:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                step=1
            )
        else:
            page_number = 1
        
        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_questions))
        page_questions = filtered_questions[start_idx:end_idx]
        
        # Display questions
        for idx, q in enumerate(page_questions):
            with st.container():
                col_q1, col_q2, col_q3 = st.columns([1, 4, 1])
                
                with col_q1:
                    st.markdown(f"**Q{q['question_number']}**")
                    st.caption(f"Year: {q['year']}")
                    st.caption(f"Diff: {q['difficulty']}")
                
                with col_q2:
                    st.markdown(f"**{q['subject']}** - {q['exam_name']}")
                    st.markdown(q['preview_text'])
                
                with col_q3:
                    # View Details button
                    if st.button("View", key=f"view_{start_idx + idx}", use_container_width=True):
                        question_details = get_question_by_id(q['question_id'])
                        if question_details:
                            display_question_details(question_details)
                    
                    # Use in Marking button
                    if st.button("Use", key=f"use_{start_idx + idx}", use_container_width=True, type="secondary"):
                        st.session_state.selected_question = get_question_by_id(q['question_id'])
                        # Switch to marking system
                        st.session_state.page = "📝 Marking System"
                        st.rerun()
                
                st.markdown("---")
        
        # Page navigation
        if total_pages > 1:
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            with col_nav2:
                st.caption(f"Page {page_number} of {total_pages}")
    else:
        st.warning("No questions match your filters. Try different search criteria.")

def display_question_details(question_data):
    """Display detailed view of a question"""
    st.subheader(f"Question Details: {question_data['question_id']}")
    
    # Metadata in columns
    col_meta1, col_meta2 = st.columns(2)
    
    with col_meta1:
        st.markdown(f"**Question Number:** {question_data['question_number']}")
        st.markdown(f"**Section:** {question_data['section']}")
        st.markdown(f"**Unit:** {question_data['unit']}")
        st.markdown(f"**AOS:** {question_data['aos']}")
        st.markdown(f"**Subtopic:** {question_data['subtopic']}")
    
    with col_meta2:
        st.markdown(f"**Skill Type:** {question_data['skill_type']}")
        st.markdown(f"**Difficulty:** {question_data['difficulty_level']}")
        st.markdown(f"**Page:** {question_data['page_number']}")
        st.markdown(f"**Answer:** {question_data['answer_text']}")
    
    st.markdown("---")
    
    # Question Text
    st.subheader("Question Text")
    st.markdown(f'<div class="question-card">{question_data["question_text"]}</div>', unsafe_allow_html=True)
    
    # Detailed Answer
    with st.expander("📘 Detailed Answer", expanded=False):
        st.markdown(question_data['detailed_answer'])
    
    # Exam Information
    with st.expander("📝 Exam Information", expanded=False):
        st.markdown(f"**Year:** {question_data['exam']['year']}")
        st.markdown(f"**Subject:** {question_data['exam']['subject']}")
        st.markdown(f"**Exam Name:** {question_data['exam']['exam_name']}")
        st.markdown(f"**Source:** {question_data['exam']['source']}")
        if question_data['exam']['pdf_url']:
            st.markdown(f"**PDF:** {question_data['exam']['pdf_url']}")
    
    # AOS Breakdown
    if question_data['aos_breakdown']:
        with st.expander("📊 AOS Breakdown", expanded=False):
            for aos in question_data['aos_breakdown']:
                st.markdown(f"- {aos['aos_name']}: {aos['percentage']}%")
    
    # Subparts
    if question_data['subparts']:
        st.subheader("Subparts")
        for subpart in question_data['subparts']:
            with st.expander(f"Subpart {subpart['subpart_letter']}", expanded=False):
                st.markdown(f"**Text:** {subpart['subpart_text']}")
                st.markdown(f"**Answer:** {subpart['subpart_answer']}")
                if subpart['subpart_detailed_answer']:
                    st.markdown(f"**Detailed Answer:** {subpart['subpart_detailed_answer']}")

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()