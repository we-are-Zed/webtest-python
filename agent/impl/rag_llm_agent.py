import os
import random
import re

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from agent.agent import Agent
from state.web_state import WebState
import paramiko
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import requests
from langchain.embeddings.base import Embeddings  # 注意 langchain 版本较新时位置在 langchain.embeddings.base


# 与大型语言模型（LLM）交互
class LLMInterface:
    def __init__(self, params):
        """
                params 中可能包含以下字段：
                - ssh_host: 服务器主机名或IP，如 'starral.sqlab.cra.moe'
                - ssh_user: SSH 用户名，如 'zyc'
                - key_filename: 私钥路径，如 '~/.ssh/id_rsa'
                - ollama_path: ollama 命令路径(如果需要自定义)
                """
        self.ssh_host = params.get("ssh_host", "starrail.sqlab.cra.moe")
        self.ssh_user = params.get("ssh_user", "zyc")
        self.key_filename = params.get("key_filename", os.path.expanduser("~/.ssh/id_rsa"))
        # 如果需要自定义ollama的可执行路径，比如 "/usr/local/bin/ollama"
        self.ollama_cmd = params.get("ollama_path", "ollama")
        self.llm_name = params.get("llm_name", "deepseek-r1:14b")

    def gpt(self, prompt: str) -> str:
        client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-vGn89b99eeaff057a82b9a50da1486c135a3b862311kQTVa"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ]
        )

        print(response.choices[0].message.content)

        return response.choices[0].message.content

    def local_llm(self, prompt: str) -> str:
        # 1. 初始化SSH连接
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(
                hostname=self.ssh_host,
                username=self.ssh_user,
                key_filename=self.key_filename
            )
            print("SSH 连接成功")

            command = f'{self.ollama_cmd} run {self.llm_name} "{prompt}"'
            command_test = f'{self.ollama_cmd} list'
            print("执行命令：", command)

            # 使用 exec_command 执行一次性命令
            stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
            exit_status = stdout.channel.recv_exit_status()  # 阻塞直到远程进程结束
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')

            if error:
                print("服务器上执行 ollama 时出错：", error)
            return output.strip()

        except Exception as e:
            print("SSH 调用失败:", e)
            return "模型调用出现错误，请稍后重试。"

        finally:
            ssh.close()


class SiliconFlowEmbeddings(Embeddings):
    """
    自定义的 Embeddings 类，用来调用 SiliconFlow API 获取文本向量。
    """

    def __init__(
            self,
            token: str,
            url: str = "https://api.siliconflow.cn/v1/embeddings",
            model: str = "BAAI/bge-large-zh-v1.5"
    ):
        """
        参数:
        - token: 你的 SiliconFlow API Token
        - url: SiliconFlow Embeddings API 的地址 (默认为 v1/embeddings)
        - model: 使用的模型名称 (默认为 'BAAI/bge-large-zh-v1.5')
        """
        super().__init__()
        self.token = token
        self.url = url
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量处理多个文档（事实上这里是循环单条请求，如果你想要批量请求，可以自行改写为一次性传 texts）
        返回一个二维列表，每个元素是对应文本的 embedding 向量。
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        处理单个查询文本的向量生成，用于例如 QA 场景下的 query embedding。
        """
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        """真正调用 SiliconFlow API 获取某个文本的向量。"""
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()  # 如果想在请求出错时抛出异常
        data = response.json()

        # 注意：以下解析逻辑要与实际返回的 JSON 对应，比如:
        # {
        #   "data": [
        #     {
        #       "embedding": [...]
        #     }
        #   ]
        # }
        #
        # 如果 SiliconFlow 的返回结构不同，请务必根据实际调整
        return data["data"][0]["embedding"]

class RetrieverInterface:
    def __init__(self, params):
        pass

    def retrieve(self, prompt: str) -> str:
        # 1. 加载数据
        loader = PyPDFLoader(r"C:\Users\ASUS\Desktop\Reasoning+RAG Web Exploration\Make LLM a Testing Expert Bringing Human-like Interaction to.pdf")

        pages = loader.load_and_split()

        # 2. 知识切片：将文档分割成均匀的块，每个块是一段原始文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = text_splitter.split_documents(pages)
        print(len(docs))

        # 3. 利用embedding模型对每个文本片段进行向量化，并储存到向量数据库中
        # embed_model = OpenAIEmbeddings(openai_api_key="")
        # vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="openai_embed")
        embed_model = SiliconFlowEmbeddings(token="sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="siliconflow_embed")

        # 4. 通过向量相似度检索和问题最相关的k个文档
        query = prompt
        result = vectorstore.similarity_search(query, k=1)

        # 5. 原始query与检索得到的文本组合起来输入语言模型，得到最终的回答
        source_knowledge = "\n".join([x.page_content for x in result])

        augmented_prompt = f"""
        Using the contexts below, answer the query.
        
        context:{source_knowledge}
        
        query: {query}
        """

        return augmented_prompt


class rag_llm_agent(Agent):
    def __init__(self, params):
        self.llm = LLMInterface(params)
        self.retriever = RetrieverInterface(params)

    def format_action_info(self, action):
        if isinstance(action, ClickAction):
            operation_name = "Click"  # 或 "ClickAction"
        elif isinstance(action, RandomInputAction):
            operation_name = "Input"  # 或 "TextInput"
        elif isinstance(action, RandomSelectAction):
            operation_name = "Select"  # 或 "SelectOption"
        else:
            operation_name = "UnknownOperation"

        widget_text = action.text if action.text else "UnnamedWidget"

        return f"Widget: '{widget_text}', can perform {operation_name}."

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        action_list = web_state.get_action_list()
        descriptions = [f"{i}. " + self.format_action_info(a) for i, a in enumerate(action_list)]

        app_name = "github"

        action_descriptions_str = "\n".join(descriptions)

        prompt = f"""
        We want to test the "{app_name}" App.

        The current page is.

        The widgets which can be operated are:
        {action_descriptions_str}

        What operation is required?
        (Operation: [click / double-click / long press / scroll / input / select] + <Widget Name>)

        please return a single number corresponding to the action index from the above list.
        If the selected action requires text input, please return the index followed by a colon and the input text.
        For example, if you choose the first action, simply return "1";
        if you choose an input action and want to provide "Hello" as input, return "2:Hello".
        
        Do not output any additional text or explanation; strictly follow the requested format.
        """.strip()

        augmented_prompt = self.retriever.retrieve(prompt)

        output = self.llm.gpt(augmented_prompt)
        print("llm返回结果：", output)

        # 解析 LLM 的输出，提取出选择的动作编号
        chosen_index = self.parse_output(output, len(action_list))
        print("选择的动作编号：", chosen_index)

        return action_list[chosen_index]

    def parse_output(self, output: str, num_actions: int) -> int:
        # 内部定义一个函数，用于去除 ANSI 转义序列
        def remove_ansi(text: str) -> str:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', text)

        # 去除思考过程
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()

        output = remove_ansi(output).strip()

        print("处理后的输出：", repr(output))

        # 匹配第一个出现的数字
        match = re.search(r"\d+", output)
        if match:
            print("解析成功，匹配到的数字：", match.group(0))
            index = int(match.group(0))
            if 0 <= index < num_actions:
                return index

        print("解析失败，随机选择一个动作。")
        return random.randint(0, num_actions - 1)


def main():
    # 用于测试 LLM 接口
    params = {}
    llm = LLMInterface(params)
    retriever = RetrieverInterface(params)
    prompt = "how to design prompt for LLM when we are using LLM for GUI test"
    augmented_prompt = retriever.retrieve(prompt)
    print("增强后的提示：", augmented_prompt)
    reasoning_output = llm.gpt(augmented_prompt)
    if "</think>" in reasoning_output:
        reasoning_output = reasoning_output.split("</think>")[-1].strip()
        print("截断后的输出：", reasoning_output)
    print("推理结果：", reasoning_output)


if __name__ == '__main__':
    main()
