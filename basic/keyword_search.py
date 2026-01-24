from rank_bm25 import BM25Okapi

corpus = [
    "paris is beutiful in summer",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "sunny paris"
tokenized_query = query.split(" ")

# Get top 1 result
print(bm25.get_top_n(tokenized_query, corpus, n=1))

#it gives an output 'It's quite windy in London' because we were looking for phrase 'windy London' and this sentance had it 
#If I change it to 'sunny paris' result is 'How is the weather today?' because in it's 'database' it had nothing much simmilar to it
#If I add phase to the 'database' 'paris is beutiful in summer' result is 'paris is beutiful in summer'