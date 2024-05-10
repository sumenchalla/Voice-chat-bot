#importing necessaary libraries
import pandas as pd
import chromadb


#Readingg data from excel fiile which contain information regarding Indain states
df=pd.read_excel("Data.xlsx")
df.columns={"Data"}

#creatind data and ids for  making vectorDB
Data = []
for i in df["Data"]:
    Data.append(str(i))
ids =[str(i+1) for i in range(len(Data))]


#Intializing the chroma client
chroma_client = chromadb.Client()
collections =chroma_client.list_collections()
collection_name ="my_collection"
#if collection_name in collections:
    # Delete the collections
#chroma_client.delete_collection(collection_name)
collection = chroma_client.create_collection(name=collection_name)
#adding Data to the collections
collection.add(documents=Data,ids=ids)
vb_collection = collection