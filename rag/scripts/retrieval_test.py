import sys
sys.path.append(".")
from rag.utils import get_paths, write_to_txt_file

import pandas as pd

from sentence_transformers import CrossEncoder


# Importing functions from your module
sys.path.append(".")
from rag.vectror_DB.faiss_search import create_embeddings, load_faiss_index, search_index


def test_with_filter(model_path, folder_path, index_name, results_file):
    write_to_txt_file(results_file, "Test Results: \n\n"+ "Embedding Model:"+ model_path+ "\n"+ "VectorDB: "+model_path+"/"+index_name+"\n")

    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cuda"
    )
    for filter_k in [3, 5, 8]:
        write_to_txt_file(results_file,"Cross Encoer: cross-encoder/ms-marco-TinyBERT-L-2-v2"+"\nk = "+str(filter_k) +"\n\n")
        
        faiss_db = load_faiss_index(folder_path=folder_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)

        # Read questions from CSV
        paths = get_paths("output/generated_questions/by_pages",".csv")
        
        write_to_txt_file(results_file,"Question Files Paths: "+ str(paths)+ "\n\n")
        
        dfs = []
        for file_path in paths:
            df = pd.read_csv(file_path)
            dfs.append(df)
        questions_df = pd.concat(dfs, ignore_index=True)
        questions = questions_df["Question"]
        context = questions_df["Context"]
        
        total_questions = len(questions)
        
        
        for k in [20]:
            write_to_txt_file(results_file, "Retrieval K = "+str(k)+"\n")
            total_count = 0
            corr_count = 0
            # Loop through questions
            for i, question in enumerate(questions):
                try:
                    # Evaluate each question (assuming they're stored as strings containing lists)
                    
                        for query in eval(question):
                            # Search the Faiss index
                            retrieved_docs = search_index(faiss_db, query, k=k)
                            reranked_docs = cross_encoder.rank(
                                query,
                                [doc.page_content for doc in retrieved_docs],
                                top_k=filter_k,
                                return_documents=True,
                            )
                            total_count += 1
                            # Check if any of the top results match the context
                            for doc in reranked_docs:
                                if doc["text"] in context[i]:
                                    corr_count+=1
                                    break
                except Exception as e:
                    print(e)
                    pass
                    
                # Update progress
                progress = (i + 1) / total_questions * 100
                print(f"Progress: {progress:.2f}%")

            
            # Print accuracy
            print("Accuracy: " , corr_count / total_count)
            print("total number of queries tested: " ,  total_count)
            write_to_txt_file(results_file,"Retrieval Accuracy: "+ str(corr_count / total_count)+ ", Total number of queries tested: " + str(total_count)+"\n\n\n\n")

def test(model_path, folder_path, index_name, results_file):
    write_to_txt_file(results_file, "Test Results: \n\n"+ "Embedding Model:"+ model_path+ "\n"+ "VectorDB: "+model_path+"/"+index_name+"\n")

    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    
    faiss_db = load_faiss_index(folder_path=folder_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)

    # Read questions from CSV
    paths = get_paths("output/generated_questions/by_pages",".csv")
    
    write_to_txt_file(results_file,"Question Files Paths: "+ str(paths)+ "\n\n")
    
    dfs = []
    for file_path in paths:
        df = pd.read_csv(file_path)
        dfs.append(df)
    questions_df = pd.concat(dfs, ignore_index=True)
    questions = questions_df["Question"]
    context = questions_df["Context"]
    
    total_questions = len(questions)
    
    
    for k in [5, 10, 15, 20, 25, 30]:
        write_to_txt_file(results_file, "Retrieval K = "+str(k)+"\n")
        total_count = 0
        corr_count = 0
        # Loop through questions
        for i, question in enumerate(questions):
            try:
                # Evaluate each question (assuming they're stored as strings containing lists)
                
                    for query in eval(question):
                        # Search the Faiss index
                        retrieved_docs = search_index(faiss_db, query, k=k)
                        total_count += 1
                        # Check if any of the top results match the context
                        for doc in retrieved_docs:
                            if doc.page_content in context[i]:
                                corr_count+=1
                                break
            except Exception as e:
                print(e)
                pass
                
            # Update progress
            progress = (i + 1) / total_questions * 100
            print(f"Progress: {progress:.2f}%")

        
        # Print accuracy
        print("Accuracy: " , corr_count / total_count)
        print("total number of queries tested: " ,  total_count)
        write_to_txt_file(results_file,"Retrieval Accuracy: "+ str(corr_count / total_count)+ ", Total number of queries tested: " + str(total_count)+"\n\n")


if __name__ == "__main__":
    test_with_filter("BAAI/bge-base-en-v1.5", "data/faiss_db/chunk-256", "FO_bge-base-en-v1.5-chunk(256)", results_file = "output/retrieval_test_results/chunk-256/bge-base_cross-encoder(k=20)_chunk(256).txt")
