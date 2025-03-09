import os
import random
import re
from action.web_action import WebAction
from agent.agent import Agent
from state.web_state import WebState
import paramiko

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

    def generate(self, prompt: str) -> str:
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

            channel = ssh.invoke_shell()

            import time
            time.sleep(1)
            while channel.recv_ready():
                initial_output = channel.recv(1024).decode('utf-8')
                print("清除初始输出:", initial_output)

            # 命令示例：ollama run <模型名称> "<prompt>"
            command = f'{self.ollama_cmd} run {self.llm_name}'
            channel.send(f'{command}\n'.encode('utf-8'))
            print("启动模型：", command)
            channel.settimeout(5.0)

            command = f'{prompt}\n'
            print("发送命令：", command)
            channel.send(f'{command}\n'.encode('utf-8'))

            output_lines = []
            while True:
                try:
                    resp = channel.recv(1024).decode('utf-8')
                    if not resp:
                        break
                    output_lines.append(resp)
                except Exception as e:
                    print("读取远程输出时出现异常:", e)
                    break

            output = "".join(output_lines)
            return output.strip()

        except Exception as e:
            print("SSH 调用失败:", e)
            return "模型调用出现错误，请稍后重试。"

        finally:
            ssh.close()

# 根据当前HTML检索相关文档
class RetrieverInterface:
    def __init__(self, params):
        # 初始化检索器，例如加载索引等
        pass

    def retrieve(self, html: str) -> str:
        # 根据html内容进行检索，返回相关文档的内容
        # 目前返回一个示例字符串
        return "检索到的相关文档示例内容。"

class rag_llm_agent(Agent):
    def __init__(self, params):
        self.llm = LLMInterface(params)

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        action_list = web_state.get_action_list()
        if not action_list:
            raise Exception("当前状态没有可执行的动作。")

        prompt = (
            "作为一位专业的移动应用测试专家，请根据以下GUI上下文信息生成下一步操作建议。\n"
            "【GUI上下文信息】\n"
            f"网页HTML: {html}\n"
            f"可执行动作列表: {action_list}\n"
            "请结合以上信息，进行功能感知的决策，并输出格式为单一的数字N，没有其他字符，其中N为建议的动作序号。\n"
            "例如，如果认为第1个动作最合适，则输出数字1。"
        )

        prompt_test = "1+1等于几？"

        reasoning_output = self.llm.generate(prompt_test)
        print("推理结果：", reasoning_output)

        # 5. 解析 LLM 的输出，提取出选择的动作编号
        chosen_index = self.parse_reasoning_output(reasoning_output, len(action_list))
        print("选择的动作编号：", chosen_index)

        return action_list[chosen_index]

    def parse_reasoning_output(self, output: str, num_actions: int) -> int:
        # 如果输出中包含 </think>，则只保留其后的内容
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()
            print("截断后的输出：", output)

        match = re.search(r"选择第(\d+)个动作", output)
        if match:
            index = int(match.group(1)) - 1  # 将1开始的编号转换为索引
            if 0 <= index < num_actions:
                return index

        print("解析失败")
        return random.randint(0, num_actions - 1)

def main():
    params = {}
    llm = LLMInterface(params)
    prompt = "1+1等于几？"
    reasoning_output = llm.generate(prompt)
    if "</think>" in reasoning_output:
        reasoning_output = reasoning_output.split("</think>")[-1].strip()
        print("截断后的输出：", reasoning_output)
    print("推理结果：", reasoning_output)



if __name__ == '__main__':
    main()