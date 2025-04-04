d= {
    "EMP001": {"Domain": "Software Development", "Efficiency": "Low"},
    "EMP002": {"Domain": "Software Development", "Efficiency": "Medium"},
    "EMP003": {"Domain": "Data Analysis", "Efficiency": "Medium"},
    "EMP004": {"Domain": "Data Analysis", "Efficiency": "Low"},
    "EMP005": {"Domain": "DevOps", "Efficiency": "High"},
    "EMP006": {"Domain": "DevOps", "Efficiency": "Low"},
    "EMP007": {"Domain": "QA Automation", "Efficiency": "Medium"},
    "EMP008": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP009": {"Domain": "Database Administration", "Efficiency": "High"},
    "EMP010": {"Domain": "Database Administration", "Efficiency": "Medium"},
    "EMP011": {"Domain": "Software Development", "Efficiency": "Low"},
    "EMP012": {"Domain": "Data Analysis", "Efficiency": "Medium"},
    "EMP013": {"Domain": "DevOps", "Efficiency": "High"},
    "EMP014": {"Domain": "QA Automation", "Efficiency": "Medium"},
    "EMP015": {"Domain": "Database Administration", "Efficiency": "Low"},
    "EMP016": {"Domain": "Software Development", "Efficiency": "Medium"},
    "EMP017": {"Domain": "Data Analysis", "Efficiency": "High"},
    "EMP018": {"Domain": "DevOps", "Efficiency": "Low"},
    "EMP019": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP020": {"Domain": "Database Administration", "Efficiency": "Low"},
    "EMP021": {"Domain": "Software Development", "Efficiency": "Medium"},
    "EMP022": {"Domain": "Data Analysis", "Efficiency": "High"},
    "EMP023": {"Domain": "DevOps", "Efficiency": "Medium"},
    "EMP024": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP025": {"Domain": "Database Administration", "Efficiency": "Medium"},
    "EMP026": {"Domain": "Software Development", "Efficiency": "Low"},
    "EMP027": {"Domain": "Data Analysis", "Efficiency": "High"},
    "EMP028": {"Domain": "DevOps", "Efficiency": "Medium"},
    "EMP029": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP030": {"Domain": "Database Administration", "Efficiency": "Medium"},
    "EMP031": {"Domain": "Software Development", "Efficiency": "Low"},
    "EMP032": {"Domain": "Data Analysis", "Efficiency": "Medium"},
    "EMP033": {"Domain": "DevOps", "Efficiency": "High"},
    "EMP034": {"Domain": "QA Automation", "Efficiency": "Medium"},
    "EMP035": {"Domain": "Database Administration", "Efficiency": "Low"},
    "EMP036": {"Domain": "Software Development", "Efficiency": "Medium"},
    "EMP037": {"Domain": "Data Analysis", "Efficiency": "High"},
    "EMP038": {"Domain": "DevOps", "Efficiency": "Medium"},
    "EMP039": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP040": {"Domain": "Database Administration", "Efficiency": "Medium"},
    "EMP041": {"Domain": "Software Development", "Efficiency": "Low"},
    "EMP042": {"Domain": "Data Analysis", "Efficiency": "Medium"},
    "EMP043": {"Domain": "DevOps", "Efficiency": "High"},
    "EMP044": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP045": {"Domain": "Database Administration", "Efficiency": "Medium"},
    "EMP046": {"Domain": "Software Development", "Efficiency": "Medium"},
    "EMP047": {"Domain": "Data Analysis", "Efficiency": "High"},
    "EMP048": {"Domain": "DevOps", "Efficiency": "Low"},
    "EMP049": {"Domain": "QA Automation", "Efficiency": "Low"},
    "EMP050": {"Domain": "Database Administration", "Efficiency": "Low"},
}
emp_id = str(input("Enter EMP ID:"))
for i in d.keys():
    if emp_id==i:
        domain = d[i]['Domain']
        eff = d[i]['Efficiency']
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#step 1:data indexing
loader  = PyPDFLoader(file_path="C://Users//MSI//Desktop//Learning Path.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text = text_splitter.split_documents(documents=data)
#step 2:embeddings 
local_embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=text,embedding=local_embedding)
#step 3:using the model to retrieve the documents
query = """Give me the learning path for {domain} domain and {eff} efficiency clusters"""
prompt = PromptTemplate(template=query,input_variables=['domain','eff'])
llm = ChatOllama(model = "llama3")
docs = vectorstore.similarity_search(query=query)
chain =RunnablePassthrough()| prompt | llm | StrOutputParser()
res = chain.invoke(input={"domain":domain, "eff":eff})
print(res)