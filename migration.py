import json
import os
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import glob
from typing import Dict, List, Any, Optional
import sys

class VCEPostgresLoader:
    def __init__(self, db_config: Dict[str, str]):
        """Initialize database connection"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("✅ Connected to PostgreSQL database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        create_tables_sql = """
        -- 1. Exams table (main exam metadata)
        CREATE TABLE IF NOT EXISTS exams (
            exam_id SERIAL PRIMARY KEY,
            year INTEGER NOT NULL,
            subject VARCHAR(100) NOT NULL,
            unit VARCHAR(50) NOT NULL,
            exam_name VARCHAR(200) NOT NULL,
            pdf_url VARCHAR(500),
            source VARCHAR(100),
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(year, subject, exam_name, source)
        );

        -- 2. AOS Breakdown table (Area of Study breakdown per exam)
        CREATE TABLE IF NOT EXISTS aos_breakdown (
            breakdown_id SERIAL PRIMARY KEY,
            exam_id INTEGER REFERENCES exams(exam_id) ON DELETE CASCADE,
            aos_name VARCHAR(200) NOT NULL,
            percentage INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 3. Questions table
        CREATE TABLE IF NOT EXISTS questions (
            question_id VARCHAR(100) PRIMARY KEY,
            exam_id INTEGER REFERENCES exams(exam_id) ON DELETE CASCADE,
            question_number INTEGER,
            section VARCHAR(10),
            unit VARCHAR(50),
            aos VARCHAR(200),
            subtopic VARCHAR(200),
            skill_type VARCHAR(100),
            difficulty_level VARCHAR(50),
            question_text TEXT,
            answer_text TEXT,
            detailed_answer TEXT,
            page_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 4. Question subparts table (if you have subquestions)
        CREATE TABLE IF NOT EXISTS question_subparts (
            subpart_id SERIAL PRIMARY KEY,
            question_id VARCHAR(100) REFERENCES questions(question_id) ON DELETE CASCADE,
            subpart_letter VARCHAR(10),
            subpart_text TEXT,
            subpart_answer TEXT,
            subpart_detailed_answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_exams_year ON exams(year);
        CREATE INDEX IF NOT EXISTS idx_exams_subject ON exams(subject);
        CREATE INDEX IF NOT EXISTS idx_questions_exam_id ON questions(exam_id);
        CREATE INDEX IF NOT EXISTS idx_questions_aos ON questions(aos);
        CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty_level);
        CREATE INDEX IF NOT EXISTS idx_aos_breakdown_exam_id ON aos_breakdown(exam_id);
        """
        
        try:
            statements = create_tables_sql.split(';')
            for statement in statements:
                if statement.strip():
                    self.cursor.execute(statement)
            self.conn.commit()
            print("✅ Tables created successfully")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error creating tables: {e}")
    
    def parse_scraped_at(self, scraped_at_str: Optional[str]) -> Optional[datetime]:
        if not scraped_at_str:
            return None
        try:
            return datetime.fromisoformat(scraped_at_str.replace('Z', '+00:00'))
        except:
            try:
                return datetime.strptime(scraped_at_str, "%Y-%m-%dT%H:%M:%SZ")
            except:
                return None
    
    def load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def insert_exam(self, exam_data: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[int]:
        try:
            exam_info = exam_data[0] if isinstance(exam_data, list) else exam_data
            scraped_at = self.parse_scraped_at(metadata.get('scraped_at'))
            
            check_sql = """
            SELECT exam_id FROM exams 
            WHERE year = %s AND subject = %s AND exam_name = %s AND source = %s
            """
            self.cursor.execute(check_sql, (
                exam_info.get('year'),
                exam_info.get('subject'),
                exam_info.get('exam'),
                metadata.get('source')
            ))
            
            existing = self.cursor.fetchone()
            if existing:
                print(f"⚠️  Exam already exists: {exam_info.get('exam')} ({exam_info.get('year')})")
                return existing[0]
            
            insert_sql = """
            INSERT INTO exams (year, subject, unit, exam_name, pdf_url, source, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING exam_id
            """
            
            self.cursor.execute(insert_sql, (
                exam_info.get('year'),
                exam_info.get('subject'),
                exam_info.get('unit'),
                exam_info.get('exam'),
                exam_info.get('pdf_url'),
                metadata.get('source'),
                scraped_at
            ))
            
            exam_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            print(f"✅ Inserted exam: {exam_info.get('exam')} ({exam_info.get('year')}) - ID: {exam_id}")
            return exam_id
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting exam: {e}")
            return None
    
    def insert_aos_breakdown(self, exam_id: int, aos_data: List[Dict[str, Any]]):
        if not aos_data:
            return
        
        try:
            insert_sql = """
            INSERT INTO aos_breakdown (exam_id, aos_name, percentage)
            VALUES (%s, %s, %s)
            """
            
            aos_records = []
            for aos_item in aos_data:
                aos_records.append((
                    exam_id,
                    aos_item.get('aos'),
                    aos_item.get('percentage')
                ))
            
            execute_batch(self.cursor, insert_sql, aos_records)
            self.conn.commit()
            print(f"✅ Inserted {len(aos_records)} AOS breakdown records")
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting AOS breakdown: {e}")
    
    def insert_questions(self, exam_id: int, questions: List[Dict[str, Any]]):
        """Insert questions data"""
        if not questions:
            return
        
        try:
            # Insert main questions
            question_sql = """
            INSERT INTO questions (
                question_id, exam_id, question_number, section, unit, aos, subtopic,
                skill_type, difficulty_level, question_text, answer_text, 
                detailed_answer, page_number
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (question_id) DO UPDATE SET
                question_text = EXCLUDED.question_text,
                answer_text = EXCLUDED.answer_text,
                detailed_answer = EXCLUDED.detailed_answer
            """
            
            question_records = []
            for question in questions:
                # Handle both string and integer question numbers
                q_number = question.get('question_number')
                if isinstance(q_number, str) and q_number.isdigit():
                    q_number = int(q_number)
                
                question_records.append((
                    question.get('question_id'),
                    exam_id,
                    q_number,
                    question.get('section'),
                    question.get('unit'),
                    question.get('aos'),
                    question.get('subtopic'),
                    question.get('skill_type'),
                    question.get('difficulty_level'),
                    question.get('question_text'),
                    question.get('answer_text'),
                    question.get('detailed_answer'),
                    question.get('page_number')
                ))
            
            execute_batch(self.cursor, question_sql, question_records)
            
            # Insert subparts if they exist
            self.insert_subparts(questions)
            
            self.conn.commit()
            print(f"✅ Inserted/Updated {len(question_records)} questions")
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting questions: {e}")
    
    def insert_subparts(self, questions: List[Dict[str, Any]]):
        """Insert question subparts if they exist"""
        subpart_records = []
        
        for question in questions:
            subparts = question.get('subparts', [])
            question_id = question.get('question_id')
            
            for subpart in subparts:
                subpart_records.append((
                    question_id,
                    subpart.get('subpart_letter'),
                    subpart.get('subpart_text'),
                    subpart.get('subpart_answer'),
                    subpart.get('subpart_detailed_answer')
                ))
        
        if subpart_records:
            try:
                subpart_sql = """
                INSERT INTO question_subparts (question_id, subpart_letter, subpart_text, 
                                               subpart_answer, subpart_detailed_answer)
                VALUES (%s, %s, %s, %s, %s)
                """
                execute_batch(self.cursor, subpart_sql, subpart_records)
                print(f"✅ Inserted {len(subpart_records)} subparts")
            except Exception as e:
                print(f"❌ Error inserting subparts: {e}")
                raise
    
    def process_json_file(self, file_path: str):
        """Process a single JSON file and insert its data"""
        print(f"\n📂 Processing: {os.path.basename(file_path)}")
        
        # Load JSON data
        data = self.load_json_file(file_path)
        if not data:
            return
        
        # Extract metadata and exams
        metadata = data.get('metadata', {})
        exams = data.get('exams', [])
        
        if not exams:
            print("⚠️  No exams found in file")
            return
        
        # Process each exam in the file
        for exam_data in exams:
            # Insert exam and get exam_id
            exam_id = self.insert_exam(exam_data, metadata)
            if not exam_id:
                continue
            
            # Insert AOS breakdown
            aos_breakdown = exam_data.get('aos_breakdown', [])
            self.insert_aos_breakdown(exam_id, aos_breakdown)
            
            # Insert questions
            questions = exam_data.get('questions', [])
            self.insert_questions(exam_id, questions)
    
    def load_all_json_files(self, directory: str):
        """Load all JSON files from a directory"""
        json_files = glob.glob(os.path.join(directory, "*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {directory}")
            return
        
        print(f"📁 Found {len(json_files)} JSON files to process")
        
        # Sort files to process in order
        json_files.sort()
        
        successful = 0
        failed = 0
        
        for json_file in json_files:
            try:
                self.process_json_file(json_file)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"❌ Failed to process {json_file}: {e}")
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Total: {len(json_files)}")
    
    def get_database_stats(self):
        """Get statistics from the database"""
        try:
            stats = {}
            
            # Count exams
            self.cursor.execute("SELECT COUNT(*) FROM exams")
            stats['total_exams'] = self.cursor.fetchone()[0]
            
            # Count questions
            self.cursor.execute("SELECT COUNT(*) FROM questions")
            stats['total_questions'] = self.cursor.fetchone()[0]
            
            # Count AOS breakdowns
            self.cursor.execute("SELECT COUNT(*) FROM aos_breakdown")
            stats['total_aos_breakdowns'] = self.cursor.fetchone()[0]
            
            # Count subparts
            self.cursor.execute("SELECT COUNT(*) FROM question_subparts")
            stats['total_subparts'] = self.cursor.fetchone()[0]
            
            # Get exam years range
            self.cursor.execute("SELECT MIN(year), MAX(year) FROM exams")
            min_year, max_year = self.cursor.fetchone()
            stats['year_range'] = f"{min_year}-{max_year}"
            
            # Get subjects list
            self.cursor.execute("SELECT DISTINCT subject FROM exams ORDER BY subject")
            subjects = [row[0] for row in self.cursor.fetchall()]
            stats['subjects'] = subjects
            
            # Get sources list
            self.cursor.execute("SELECT DISTINCT source FROM exams ORDER BY source")
            sources = [row[0] for row in self.cursor.fetchall()]
            stats['sources'] = sources
            
            print("\n📊 Database Statistics:")
            print(f"   Exams: {stats['total_exams']}")
            print(f"   Questions: {stats['total_questions']}")
            print(f"   AOS Breakdowns: {stats['total_aos_breakdowns']}")
            print(f"   Subparts: {stats['total_subparts']}")
            print(f"   Year Range: {stats['year_range']}")
            print(f"   Subjects: {', '.join(stats['subjects'])}")
            print(f"   Sources: {', '.join(stats['sources'])}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return None


def main():    
    db_config = {
        'host': 'localhost',           # or your PostgreSQL host
        'database': 'vce_learning_platform',
        'user': 'postgres',            # your PostgreSQL username
        'password': 'postgres1234',        # your PostgreSQL password
        'port': '5432'                 # default PostgreSQL port
    }
    
    json_directory = r"E:\Practice\VCE-Learning-Plateform\outputs"
    
    loader = VCEPostgresLoader(db_config)
    
    try:
        loader.connect()
        loader.create_tables()
        loader.load_all_json_files(json_directory)
        loader.get_database_stats()
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
    finally:
        loader.close()


if __name__ == "__main__":
    # pip install psycopg2-binary
    main()