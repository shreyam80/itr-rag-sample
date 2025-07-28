# Section 1: Data Loading
from datasets import load_dataset
import pandas as pd
import re

# Load the coding dataset - it's already structured, not XML!
dataset = load_dataset("ZorraZabb/full_coding_sampling_xml_fitered", split=
"train[:10000]")

# Explore the actual dataset structure
print("Dataset fields:", dataset.column_names)
print("Sample record:", dataset[0])

def analyze_dataset_structure(dataset):
    """Analyze the dataset structure and content"""
    # Convert to pandas for easier analysis
    df = dataset.to_pandas()

    print("Dataset shape:", df.shape)
    print("Languages distribution:")
    print(df['lang'].value_counts())
    
    print("Repository popularity (top 10):")
    print(df.nlargest(10,'star')[['repo_name','star','lang']])

    print("Average token length by language:")
    print(df.groupby('lang')['len_tokens'].mean().sort_values(ascending=False))
    return df

def preprocess_code_sample(sample):
    """Preprocess individual code samples"""
    return {
        'code': sample['text'],
        'language': sample['lang'],
        'file_path': sample['dir'],
        'repo_name': sample['repo_name'],
        'repo_full_name': sample['repo_full_name'],
        'star_count': sample['star'],
        'token_count': sample['len_tokens'],
        'created_date': sample['created_date'],
        'updated_date': sample['updated_date']
    }

def filter_quality_code(df):
    """Filter for high-quality code samples"""
    # Filter criteria
    quality_filters = (
        (df['len_tokens'] >= 50) & # Minimum meaningful code length
        (df['len_tokens'] <= 2000) & # Not too long for processing
        (df['star'] >= 10) & # From popular repositories
        (df['lang'].isin(['python','javascript','java','cpp','c','objective-c']))
    )
    return df[quality_filters]

import ast
from tree_sitter import Language, Parser

class CodeProcessor:
    def intelligent_code_chunking(self, code_sample, max_tokens=800):
        """Chunk code based on natural boundaries and repository context"""
        code_text = code_sample['code']
        language = code_sample['language']
        file_path = code_sample['file_path']
        
        # Use file extension and language info for better chunking
        if language == 'python':
            return self.chunk_python_code(code_text, max_tokens)
        elif language == 'javascript':
            return self.chunk_javascript_code(code_text, max_tokens)
        elif language in ['objective-c', 'c', 'cpp']:
            return self.chunk_c_family_code(code_text, max_tokens)
        else:
            return self.semantic_code_split(code_text, max_tokens)
    
    def extract_enhanced_metadata(self, code_sample):
        """Extract comprehensive metadata using dataset fields"""
        code_text = code_sample['code']
        
        return {
            'language': code_sample['language'],
            'repo_name': code_sample['repo_name'],
            'repo_popularity': code_sample['star_count'],
            'file_path': code_sample['file_path'],
            'token_count': code_sample['token_count'],
            'functions': self.extract_functions(code_text, code_sample['language']),
            'imports': self.extract_imports(code_text, code_sample['language']),
            'complexity_score': self.calculate_complexity(code_text),
            'code_type': self.classify_code_type(code_text, code_sample['file_path']),
            'repo_context': self.extract_repo_context(code_sample),
            'last_updated': code_sample['updated_date']
        }
    
    def classify_code_type(self, code_text, file_path):
        """Classify code type based on content and file path"""
        if file_path.endswith('.h'):
            return 'header_file'
        elif file_path.endswith(('.py', '.js', '.java')):
            if 'class' in code_text.lower():
                return 'class_definition'
            elif 'function' in code_text.lower() or 'def ' in code_text:
                return 'function_implementation'
            elif 'import' in code_text.lower() or 'require(' in code_text:
                return 'module_with_imports'
        return 'code_snippet'
    
    def extract_repo_context(self, code_sample):
        """Extract repository context for better understanding"""
        return {
            'repo_name': code_sample['repo_name'],
            'popularity': code_sample['star_count'],
            'likely_domain': self.infer_domain_from_repo(code_sample['repo_name']),
            'file_type': code_sample['file_path'].split('.')[-1] if '.' in code_sample['file_path'] else 'unknown'
        }
    
    def infer_domain_from_repo(self, repo_name):
        """Infer application domain from repository name"""
        domain_keywords = {
            'web': ['web', 'http', 'server', 'api', 'express'],
            'mobile': ['ios', 'android', 'mobile', 'app'],
            'security': ['hack', 'security', 'vuln', 'exploit'],
            'ui': ['ui', 'controller', 'view', 'pager'],
            'automation': ['auto', 'script', 'bot', 'tool']
        }
        
        repo_lower = repo_name.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in repo_lower for keyword in keywords):
                return domain
        return 'general'
    import chromadb
from sentence_transformers import SentenceTransformer

class CodeVectorStore:
    def __init__(self):
        self.code_embedder = SentenceTransformer('microsoft/codebert-base')
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collections = {
            'code_snippets': self.client.create_collection('code_snippets'),
            'documentation': self.client.create_collection('documentation'),
            'examples': self.client.create_collection('examples')
        }
    
    def add_code_documents(self, processed_chunks):
        for chunk in processed_chunks:
            code_embedding = self.code_embedder.encode(chunk['code'])
            collection = self.collections['code_snippets']
            
            collection.add(
                embeddings=[code_embedding.tolist()],
                documents=[chunk['code']],
                metadatas=[chunk['metadata']],
                ids=[f"code_{len(collection.get()['ids'])}"]
            )
class RepositoryAwareRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def repository_aware_search(self, query, filters=None, k=10):
        """Search with repository context and popularity weighting"""
        # Basic semantic search
        semantic_results = self.semantic_search(query, filters, k*2)
        
        # Apply repository popularity weighting
        weighted_results = self.apply_popularity_weighting(semantic_results)
        
        # Filter by repository quality metrics
        quality_filtered = self.filter_by_quality(weighted_results)
        
        return quality_filtered[:k]
    
    def apply_popularity_weighting(self, results):
        """Weight results by repository popularity"""
        for result in results:
            star_count = result['metadata'].get('repo_popularity', 0)
            popularity_boost = min(star_count / 1000, 2.0)  # Cap boost at 2x
            result['weighted_score'] = result['distance'] * (1 + popularity_boost)
        
        return sorted(results, key=lambda x: x['weighted_score'])
    
    def filter_by_quality(self, results):
        """Filter results based on code quality indicators"""
        quality_results = []
        for result in results:
            metadata = result['metadata']
            
            # Quality filters
            if (metadata.get('token_count', 0) >= 50 and 
                metadata.get('repo_popularity', 0) >= 10 and
                metadata.get('complexity_score', 0) <= 8):
                quality_results.append(result)
        
        return quality_results
    
    def domain_specific_search(self, query, domain_filter=None):
        """Search within specific domains (web, mobile, security, etc.)"""
        if domain_filter:
            # Filter by inferred domain from repository context
            domain_results = []
            all_results = self.semantic_search(query, None, 50)
            
            for result in all_results:
                repo_domain = result['metadata'].get('repo_context', {}).get('likely_domain')
                if repo_domain == domain_filter:
                    domain_results.append(result)
            
            return domain_results[:10]
        
        return self.repository_aware_search(query, None, 10)
    
#query enhancement
def expand_programming_query(self, query):
    """Enhance queries with programming synonyms"""
    # Add technical synonyms
    synonyms = {
        'function': ['method', 'procedure', 'routine'],
        'array': ['list', 'collection', 'sequence'],
        'loop': ['iteration', 'cycle', 'repeat']
    }
    # Implementation...

def section_aware_retrieval(self, query):
    """Route queries to appropriate code sections"""
    if 'implement' in query.lower():
        return self.search_in_collections(['examples', 'code_snippets'])
    elif 'explain' in query.lower():
        return self.search_in_collections(['documentation'])
    # More routing logic...

#LLM Integration
import ollama

class CodeLLMIntegrator:
    def __init__(self, model_name="codellama:7b"):
        self.model_name = model_name
        ollama.pull(model_name)
    
    def generate_code_answer(self, query, retrieved_code):
        """Generate comprehensive coding assistance"""
        context = self.build_code_context(retrieved_code)
        prompt_template = self.select_prompt_template(query)
        
        prompt = prompt_template.format(
            query=query,
            context=context,
            examples=self.format_code_examples(retrieved_code[:3])
        )
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={'temperature': 0.3, 'top_p': 0.9}
        )
        return response['response']

    CODE_GENERATION_PROMPT = """
You are an expert programming assistant. 

Query: {query}
Context: {context}
Examples: {examples}

Provide working code with explanations and best practices.
"""

#context enhancement - skipped

# evaluation framework
class CodeRAGEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.test_queries = [
            # Python queries
            "How to implement WeChat automation in Python?",
            "Python socket programming examples",
            "How to handle XML parsing in Python?",
            
            # JavaScript queries  
            "Express.js server setup with routes",
            "Browser fingerprinting techniques in JavaScript",
            "How to detect user zoom level in browser?",
            
            # Security-related queries
            "XSS polyglot examples for penetration testing",
            "Network reconnaissance automation scripts",
            "How to implement deanonymization attacks?",
            
            # Mobile development
            "iOS UIViewController examples in Objective-C",
            "How to create pager controllers for iOS?",
            
            # Repository-specific queries
            "Code from popular GitHub repositories (>1000 stars)",
            "Recent code updates from 2024"
        ]
    
    def evaluate_retrieval_accuracy(self):
        """Test retrieval quality with actual dataset characteristics"""
        results = {}
        
        for query in self.test_queries:
            retrieved = self.rag_system.search(query, k=10)
            
            # Evaluate based on actual dataset metadata
            relevance_score = self.calculate_relevance(query, retrieved)
            popularity_score = self.calculate_popularity_score(retrieved)
            diversity_score = self.calculate_language_diversity(retrieved)
            
            results[query] = {
                'relevance': relevance_score,
                'popularity': popularity_score,
                'diversity': diversity_score,
                'total_results': len(retrieved)
            }
        
        return results
    
    def calculate_popularity_score(self, results):
        """Calculate average repository popularity of results"""
        if not results:
            return 0
        
        star_counts = [r['metadata'].get('repo_popularity', 0) for r in results]
        return sum(star_counts) / len(star_counts)
    
    def calculate_language_diversity(self, results):
        """Calculate programming language diversity in results"""
        languages = set(r['metadata'].get('language') for r in results)
        return len(languages)
    
    def evaluate_code_generation_quality(self):
        """Test generated code using actual examples from dataset"""
        # Use actual popular code samples as ground truth
        test_cases = [
            {
                'query': 'Create an Express.js server with basic routing',
                'expected_patterns': ['express', 'app.get', 'app.listen', 'require'],
                'language': 'javascript'
            },
            {
                'query': 'Python script for network reconnaissance',
                'expected_patterns': ['socket', 'subprocess', 'nmap', 'whois'],
                'language': 'python'
            }
        ]
        
        for test_case in test_cases:
            generated_code = self.rag_system.generate_code(test_case['query'])
            quality_score = self.assess_generated_code_quality(
                generated_code, 
                test_case['expected_patterns'],
                test_case['language']
            )
            print(f"Query: {test_case['query']}")
            print(f"Quality Score: {quality_score}/10")
    
    def benchmark_realistic_performance(self):
        """Benchmark with realistic dataset characteristics"""
        # Test with different repository popularity levels
        popularity_filters = [
            ('popular', 1000),  # 1000+ stars
            ('moderate', 100),  # 100+ stars  
            ('all', 0)          # All repositories
        ]
        
        performance_results = {}
        
        for filter_name, min_stars in popularity_filters:
            start_time = time.time()
            
            results = self.rag_system.search(
                "JavaScript automation examples",
                filters={'min_stars': min_stars},
                k=20
            )
            
            end_time = time.time()
            
            performance_results[filter_name] = {
                'response_time': end_time - start_time,
                'result_count': len(results),
                'avg_repo_popularity': self.calculate_popularity_score(results)
            }
        
        return performance_results