import os
import random
import re
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from agent.agent import Agent
from state.web_state import WebState
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 处理Chroma版本兼容性
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain.embeddings.base import Embeddings
from langchain.schema import Document


# 与大型语言模型（LLM）交互 - 使用SiliconFlow API
class LLMInterface:
    def __init__(self, params):
        """
        使用SiliconFlow API进行LLM交互
        """
        self.api_key = params.get("api_key", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.chat_url = params.get("chat_url", "https://api.siliconflow.cn/v1/chat/completions")
        self.model = params.get("model", "deepseek-ai/DeepSeek-R1")
        self.enable_thinking = params.get("enable_thinking", False)
        self.max_tokens = params.get("max_tokens", 1024)
        self.temperature = params.get("temperature", 0.7)
        
        # 🆔 为每个LLM实例生成唯一会话标识符
        import uuid
        self.session_id = str(uuid.uuid4())
        self.test_session_counter = 0  # 测试会话计数器
        
        print(f"🆔 LLM会话ID: {self.session_id}")

    def reset_session(self):
        """
        重置会话状态 - 为新测试创建完全独立的会话
        """
        import uuid
        old_session_id = self.session_id
        self.session_id = str(uuid.uuid4())
        self.test_session_counter += 1
        
        print(f"🔄 LLM会话重置:")
        print(f"  旧会话ID: {old_session_id}")
        print(f"  新会话ID: {self.session_id}")
        print(f"  测试计数: {self.test_session_counter}")

    def _build_isolation_prompt(self, user_prompt: str) -> str:
        """
        构建包含隔离信息的提示词，确保测试独立性
        """
        isolation_header = f"""
[🔒 测试隔离声明]
- 这是一个全新的独立测试会话
- 会话ID: {self.session_id}
- 测试编号: {self.test_session_counter}
- 请忽略任何之前的对话历史和记忆
- 请基于当前提供的信息独立做出决策
- 不要参考之前测试的结果或经验

[📋 当前测试任务]
{user_prompt}

请严格基于上述信息进行推理和决策，确保测试的独立性和一致性。
"""
        return isolation_header.strip()
        
    def chat_with_thinking(self, prompt: str) -> Dict[str, str]:
        """
        发送聊天请求并返回内容和thinking过程
        返回: {"content": "回答内容", "reasoning": "思考过程"}
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # 🆔 添加会话标识符到请求头
            "X-Session-ID": self.session_id,
            "X-Test-Session": str(self.test_session_counter)
        }
        
        # 🔒 构建包含隔离信息的提示词
        isolated_prompt = self._build_isolation_prompt(prompt)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"你是一个Web测试专家。当前会话ID: {self.session_id}。这是一个独立的测试会话，请忽略任何之前的对话记忆。"
                },
                {
                    "role": "user", 
                    "content": isolated_prompt
                }
            ],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            # 🎲 添加随机种子确保每次调用的独立性
            "seed": hash(self.session_id + str(self.test_session_counter)) % 2147483647
        }
        
        # 只有支持thinking的模型才添加相关参数
        if "QwQ" in self.model and self.enable_thinking:
            payload.update({
                "enable_thinking": True,
                "thinking_budget": 4096
            })
        
        try:
            response = requests.post(self.chat_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            
            print(f"LLM响应 [会话:{self.session_id[:8]}]: {content}")
            if reasoning:
                print(f"LLM推理过程 [会话:{self.session_id[:8]}]: {reasoning[:200]}...")  # 只打印前200个字符
                
            return {
                "content": content,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"LLM调用失败 [会话:{self.session_id[:8]}]: {e}")
            return {
                "content": "模型调用出现错误，请稍后重试。",
                "reasoning": ""
            }


class SiliconFlowEmbeddings(Embeddings):
    """
    使用SiliconFlow API的嵌入服务
    """

    def __init__(
            self,
            token: str = "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
            url: str = "https://api.siliconflow.cn/v1/embeddings",
            model: str = "BAAI/bge-large-zh-v1.5",
            verbose: bool = False
    ):
        super().__init__()
        self.token = token
        self.url = url
        self.model = model
        self.verbose = verbose

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量处理多个文档的向量化"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """处理单个查询文本的向量化"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """调用SiliconFlow API获取文本向量"""
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            if self.verbose:
                print(f"嵌入向量获取失败: {e}")
            # 返回一个默认的向量（全零向量，维度假设为1024）
            return [0.0] * 1024


class StateKnowledgeBase:
    """
    专门管理页面状态信息的知识库
    存储每个测试状态的详细信息：可点击组件、页面结构、测试路径等
    """
    
    def __init__(self, params, verbose=False):
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("state_collection_name", "state_knowledge")
        self.chunk_size = params.get("state_chunk_size", 800)  # 状态信息较大，使用更大的chunk
        self.chunk_overlap = params.get("state_chunk_overlap", 100)
        self.max_entries = params.get("max_state_entries", 500)
        self.persist_directory = params.get("state_persist_directory", "./state_vectorstore")
        self.verbose = verbose
        
        # 初始化嵌入模型和向量存储
        self.embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=verbose)
        self.vectorstore = None  # 延迟初始化
        
        # 如果需要清空state知识库
        if params.get("clear_state_on_init", True):
            self.clear_state_vectorstore()
    
    def _initialize_vectorstore(self):
        """初始化或加载已有的向量存储"""
        if self.vectorstore is not None:
            return
            
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print(f"Loaded state KB with {self.vectorstore._collection.count()} documents")
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print("Created new state KB")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize state KB: {e}")
            self.vectorstore = Chroma(
                embedding_function=self.embed_model,
                collection_name=self.collection_name
            )

    def clear_state_vectorstore(self):
        """清空状态知识库"""
        if self.verbose:
            print("Clearing state KB...")
        
        # 关闭现有连接
        if self.vectorstore is not None:
            try:
                if hasattr(self.vectorstore, '_client') and self.vectorstore._client:
                    self.vectorstore._client.reset()
                if hasattr(self.vectorstore, '_collection'):
                    del self.vectorstore._collection
                del self.vectorstore
            except Exception as e:
                if self.verbose:
                    print(f"Error closing state vectorstore: {e}")
            finally:
                self.vectorstore = None
        
        import gc
        gc.collect()
        
        # 删除持久化目录
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("State KB cleared")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to clear state KB: {e}")
        
        # 重新创建空目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to create state directory: {e}")
    
    def add_page_state(self, page_title: str, action_list: List, selected_action_index: int, 
                      reasoning: str = "", timestamp: str = None):
        """
        添加详细的页面状态信息到知识库
        """
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            # 详细分析可用动作
            click_actions = []
            input_actions = []
            select_actions = []
            other_actions = []
            
            for i, action in enumerate(action_list):
                action_info = {
                    "index": i,
                    "text": getattr(action, 'text', 'Unknown'),
                    "type": type(action).__name__
                }
                
                if isinstance(action, ClickAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'unknown'),
                        "addition_info": getattr(action, 'addition_info', '')
                    })
                    click_actions.append(action_info)
                elif isinstance(action, RandomInputAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'input')
                    })
                    input_actions.append(action_info)
                elif isinstance(action, RandomSelectAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'select'),
                        "options": getattr(action, 'options', 'N/A')
                    })
                    select_actions.append(action_info)
                else:
                    other_actions.append(action_info)
            
            # 构建结构化的状态信息
            state_summary = {
                "total_actions": len(action_list),
                "click_actions_count": len(click_actions),
                "input_actions_count": len(input_actions),
                "select_actions_count": len(select_actions),
                "other_actions_count": len(other_actions),
                "selected_action_index": selected_action_index,
                "selected_action_type": type(action_list[selected_action_index]).__name__ if selected_action_index < len(action_list) else "Invalid"
            }
            
            # 构造详细的文档内容
            content = f"""
时间: {timestamp}
页面标题: {page_title}

页面状态概览:
- 总可用动作数: {state_summary['total_actions']}
- 可点击元素: {state_summary['click_actions_count']} 个
- 输入字段: {state_summary['input_actions_count']} 个  
- 选择框: {state_summary['select_actions_count']} 个
- 其他动作: {state_summary['other_actions_count']} 个

选择的动作:
- 索引: {selected_action_index}
- 类型: {state_summary['selected_action_type']}
- 详情: {getattr(action_list[selected_action_index], 'text', 'Unknown') if selected_action_index < len(action_list) else 'Invalid'}

详细可点击元素:
{json.dumps(click_actions, ensure_ascii=False, indent=2)}

详细输入字段:
{json.dumps(input_actions, ensure_ascii=False, indent=2)}

详细选择框:
{json.dumps(select_actions, ensure_ascii=False, indent=2)}

决策推理:
{reasoning[:500]}{'...' if len(reasoning) > 500 else ''}

测试路径分析:
页面 "{page_title}" 提供了 {state_summary['total_actions']} 个交互选项，主要包含 {state_summary['click_actions_count']} 个点击目标和 {state_summary['input_actions_count']} 个输入机会。这种配置适合进行{'导航测试' if state_summary['click_actions_count'] > state_summary['input_actions_count'] else '表单测试'}。
"""
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "timestamp": timestamp,
                    "page_title": page_title[:100],
                    "total_actions": state_summary['total_actions'],
                    "click_actions": state_summary['click_actions_count'],
                    "input_actions": state_summary['input_actions_count'],
                    "selected_action_index": selected_action_index,
                    "selected_action_type": state_summary['selected_action_type'],
                    "type": "page_state_record"
                }
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            
            # 持久化
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except Exception:
                pass
            
            if self.verbose:
                print(f"Saved state: {page_title} ({state_summary['total_actions']} actions, {len(split_docs)} chunks)")
            
            # 清理旧记录
            self._cleanup_old_records()
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save page state: {e}")
    
    def _cleanup_old_records(self):
        """清理过多的旧记录，保持知识库在合理大小"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, '_collection'):
                current_count = self.vectorstore._collection.count()
                if current_count > self.max_entries:
                    if self.verbose:
                        print(f"Warning: State KB has {current_count} records (limit: {self.max_entries})")
        except Exception as e:
            if self.verbose:
                print(f"Error checking state records: {e}")
    
    def retrieve_similar_states(self, current_page_title: str, action_count: int, k: int = 3) -> str:
        """
        检索相似的页面状态信息
        """
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return ""
            
            # 构建查询
            query = f"页面标题 {current_page_title} 动作数量 {action_count} 页面状态 交互选项"
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                return ""
            
            # 组织检索到的状态记录
            similar_states = []
            for doc in results:
                similar_states.append(doc.page_content)
            
            state_context = "\n--- 相似页面状态参考 ---\n" + "\n\n".join(similar_states)
            return state_context
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve similar states: {e}")
            return ""


class ThinkingKnowledgeBase:
    """
    管理thinking过程的知识库
    """
    
    def __init__(self, params, verbose=False):
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("thinking_collection_name", "thinking_knowledge")
        self.chunk_size = params.get("thinking_chunk_size", 500)
        self.chunk_overlap = params.get("thinking_chunk_overlap", 50)
        self.max_entries = params.get("max_thinking_entries", 1000)  # 限制知识库大小
        self.persist_directory = params.get("thinking_persist_directory", "./thinking_vectorstore")
        self.verbose = verbose
        
        # 初始化嵌入模型和向量存储
        self.embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=verbose)
        self.vectorstore = None  # 延迟初始化
        
        # 如果需要清空thinking知识库
        if params.get("clear_thinking_on_init", True):
            self.clear_thinking_vectorstore()
        
    def _initialize_vectorstore(self):
        """初始化或加载已有的向量存储"""
        if self.vectorstore is not None:
            return
            
        try:
            # 尝试加载现有的向量存储
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print(f"Loaded thinking KB with {self.vectorstore._collection.count()} documents")
            else:
                # 创建新的向量存储
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print("Created new thinking KB")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize thinking KB: {e}")
            # 创建临时的内存向量存储作为备用
            self.vectorstore = Chroma(
                embedding_function=self.embed_model,
                collection_name=self.collection_name
            )

    def clear_thinking_vectorstore(self):
        """
        清空Thinking知识库 - 彻底删除所有相关文件
        """
        if self.verbose:
            print("Clearing thinking KB...")
        
        # 1. 关闭现有的向量存储连接
        if self.vectorstore is not None:
            try:
                # 尝试关闭Chroma连接
                if hasattr(self.vectorstore, '_client') and self.vectorstore._client:
                    self.vectorstore._client.reset()
                if hasattr(self.vectorstore, '_collection'):
                    del self.vectorstore._collection
                del self.vectorstore
            except Exception as e:
                if self.verbose:
                    print(f"Error closing thinking vectorstore: {e}")
            finally:
                self.vectorstore = None
        
        # 2. 强制垃圾回收
        import gc
        gc.collect()
        
        # 3. 删除持久化目录
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("Thinking KB cleared")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to clear thinking KB: {e}")
        
        # 4. 重新创建空目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to create thinking directory: {e}")
    
    def add_thinking(self, prompt: str, reasoning: str, action_taken: str, timestamp: str = None):
        """
        将thinking过程添加到知识库
        """
        if not reasoning or reasoning.strip() == "":
            return
            
        try:
            # 延迟初始化向量存储
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            # 构造文档内容，包含上下文信息
            content = f"""
            时间: {timestamp}
            测试场景: {prompt[:200]}...
            推理过程: {reasoning}
            采取的行动: {action_taken}
            """
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "timestamp": timestamp,
                    "prompt_preview": prompt[:100],
                    "action": action_taken,
                    "type": "thinking_record"
                }
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents([doc])

            self.vectorstore.add_documents(split_docs)
            
            # 兼容旧版本的持久化方法
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except Exception:
                pass  # 新版本自动持久化，忽略错误
            
            if self.verbose:
                print(f"Saved thinking: {action_taken[:30]}... ({len(split_docs)} chunks)")
            
            # 检查并清理旧记录以控制大小
            self._cleanup_old_records()
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save thinking: {e}")
    
    def _cleanup_old_records(self):
        """清理过多的旧记录，保持知识库在合理大小"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, '_collection'):
                current_count = self.vectorstore._collection.count()
                if current_count > self.max_entries:
                    # 这里可以实现更复杂的清理策略，比如删除最旧的记录
                    # 当前简单实现：当记录过多时给出警告
                    if self.verbose:
                        print(f"Warning: Thinking KB has {current_count} records (limit: {self.max_entries})")
        except Exception as e:
            if self.verbose:
                print(f"Error checking thinking records: {e}")
    
    def retrieve_relevant_thinking(self, query: str, k: int = 3) -> str:
        """
        根据查询检索相关的thinking记录
        """
        try:
            # 延迟初始化向量存储
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return ""
            
            # 搜索相关文档
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                return ""
            
            # 组织检索到的thinking记录
            relevant_thinking = []
            for doc in results:
                relevant_thinking.append(doc.page_content)
            
            thinking_context = "\n--- 相关的历史推理经验 ---\n" + "\n\n".join(relevant_thinking)
            return thinking_context
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve thinking: {e}")
            return ""


class RetrieverInterface:
    def __init__(self, params, verbose=False):
        self.knowledge_path = params.get("knowledge_path", r"C:\Users\ASUS\Desktop\Reasoning+RAG Web Exploration\Make LLM a Testing Expert Bringing Human-like Interaction to.pdf")
        
        # 固定的网页测试注意事项PDF文件路径 - 修复路径计算
        # 获取当前脚本的目录，然后计算相对于项目根目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # 向上两级到项目根目录
        self.web_testing_pdf = os.path.join(project_root, "agent", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf")
        
        self.chunk_size = params.get("chunk_size", 500)
        self.chunk_overlap = params.get("chunk_overlap", 50)
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("collection_name", "siliconflow_embed")
        self.top_k = params.get("top_k", 3)
        self.verbose = verbose
        
        # 设置持久化目录 - 用于RAG数据库存储
        self.persist_directory = params.get("rag_persist_directory", "./rag_vectorstore")
        
        # 加载文档和创建向量库的缓存
        self._vectorstore = None

    def _close_vectorstore_connections(self):
        """
        彻底关闭向量存储连接和释放资源
        """
        if self._vectorstore is not None:
            try:
                if self.verbose:
                    print("Closing vectorstore connections...")
                
                # 关闭Chroma客户端连接
                if hasattr(self._vectorstore, '_client') and self._vectorstore._client:
                    try:
                        self._vectorstore._client.reset()
                    except Exception as e:
                        if self.verbose:
                            print(f"Failed to reset Chroma client: {e}")
                
                # 清理集合引用
                if hasattr(self._vectorstore, '_collection'):
                    try:
                        del self._vectorstore._collection
                    except Exception as e:
                        if self.verbose:
                            print(f"Failed to delete collection: {e}")
                
                # 删除向量存储对象
                del self._vectorstore
                
            except Exception as e:
                if self.verbose:
                    print(f"Error closing vectorstore: {e}")
            finally:
                self._vectorstore = None

    def _force_close_sqlite_connections(self):
        """
        强制关闭SQLite连接（改进版）
        """
        try:
            import sqlite3
            import glob
            
            if self.verbose:
                print("Forcing SQLite connections to close...")
            
            # 查找所有相关的SQLite文件
            sqlite_patterns = [
                os.path.join(self.persist_directory, "**/*.sqlite*"),
                os.path.join(self.persist_directory, "**/chroma*"),
            ]
            
            sqlite_files = []
            for pattern in sqlite_patterns:
                sqlite_files.extend(glob.glob(pattern, recursive=True))
            
            if sqlite_files and self.verbose:
                print(f"Found {len(sqlite_files)} database files")
            
            # 等待一段时间让连接自然关闭
            import time
            time.sleep(0.5)
            
        except Exception as e:
            if self.verbose:
                print(f"Error forcing SQLite connections close: {e}")

    def _remove_files_with_retry(self, max_attempts=5, delay=1):
        """
        带重试机制的文件删除（改进版）
        """
        for attempt in range(max_attempts):
            try:
                if not os.path.exists(self.persist_directory):
                    return True
                
                if self.verbose:
                    print(f"Attempting to remove directory (attempt {attempt + 1}/{max_attempts})")
                
                # 删除目录
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("RAG vectorstore cleared successfully")
                return True
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    if self.verbose:
                        print(f"Deletion failed, retrying in {delay}s... Error: {e}")
                    import time
                    time.sleep(delay)
                    delay *= 1.5  # 指数退避
                else:
                    if self.verbose:
                        print(f"Failed to delete directory after {max_attempts} attempts: {e}")
                    return False
        
        return False

    def clear_vectorstore(self):
        """
        彻底清空RAG数据库 - 完全改进版
        """
        if self.verbose:
            print("Clearing RAG database...")
        
        # 步骤1: 关闭所有向量存储连接
        self._close_vectorstore_connections()
        
        # 步骤2: 强制垃圾回收
        import gc
        gc.collect()
        
        # 步骤3: 强制关闭SQLite连接
        self._force_close_sqlite_connections()
        
        # 步骤4: 带重试机制的文件删除
        success = self._remove_files_with_retry()
        
        # 步骤5: 重新创建空的持久化目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 验证目录确实为空
            files_remaining = []
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    files_remaining.extend(files)
            
            if not files_remaining:
                if self.verbose:
                    print("RAG database cleared completely")
            else:
                if self.verbose:
                    print(f"Warning: {len(files_remaining)} files still remain")
                    
        except Exception as e:
            if self.verbose:
                print(f"Failed to create persist directory: {e}")

    def _load_vectorstore(self):
        """延迟加载向量存储，避免不必要的资源消耗"""
        if self._vectorstore is not None:
            return

        if self.verbose:
            print("Initializing RAG knowledge base...")

        # 添加备用路径检查
        possible_paths = [
            self.web_testing_pdf,
            os.path.join(os.getcwd(), "..", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf"),
            os.path.join(os.path.dirname(os.getcwd()), "agent", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf"),
            "../../agent/网页测试核心注意事项 (Core Considerations for Web Testing).pdf"
        ]

        actual_pdf_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                actual_pdf_path = abs_path
                if self.verbose:
                    print(f"Found PDF: {os.path.basename(abs_path)}")
                break
        
        try:
            all_docs = []
            
            # 1. 加载原有的知识文件（如果存在）
            if self.knowledge_path and os.path.exists(self.knowledge_path):
                if self.verbose:
                    print(f"Loading knowledge file: {os.path.basename(self.knowledge_path)}")
                loader = PyPDFLoader(self.knowledge_path)
                pages = loader.load_and_split()
                
                # 为文档添加来源标记
                for page in pages:
                    page.metadata["source_file"] = "Knowledge Base"
                
                all_docs.extend(pages)
                if self.verbose:
                    print(f"Loaded {len(pages)} pages")
            
            # 2. 加载固定的网页测试注意事项PDF
            if actual_pdf_path:
                if self.verbose:
                    print("Loading web testing guidelines")
                loader = PyPDFLoader(actual_pdf_path)
                pages = loader.load_and_split()
                
                # 为文档添加来源标记
                for page in pages:
                    page.metadata["source_file"] = "Web Testing Guidelines"
                
                all_docs.extend(pages)
                if self.verbose:
                    print(f"Loaded {len(pages)} pages")
            else:
                if self.verbose:
                    print("Warning: Web testing guidelines PDF not found")
            
            if not all_docs:
                if self.verbose:
                    print("Warning: No knowledge documents found")
                # 创建空的向量存储
                embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
                self._vectorstore = Chroma(
                    embedding_function=embed_model,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                return
            
            # 4. 知识切片
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            docs = text_splitter.split_documents(all_docs)
            if self.verbose:
                print(f"Document splitting complete: {len(docs)} chunks")
            
            # 5. 创建向量数据库（使用持久化存储）
            embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
            self._vectorstore = Chroma.from_documents(
                documents=docs, 
                embedding=embed_model, 
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            if self.verbose:
                print("RAG knowledge base initialized successfully")
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load vectorstore: {e}")
            # 创建空的向量存储以避免后续错误
            embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
            self._vectorstore = Chroma(
                embedding_function=embed_model,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )

    def retrieve(self, prompt: str) -> str:
        try:
            # 延迟初始化向量库
            if not self._vectorstore:
                self._load_vectorstore()
            
            if not self._vectorstore:
                if self.verbose:
                    print("Vectorstore not initialized, cannot retrieve")
                return prompt
                
            # 检索相关文档
            query = prompt
            result = self._vectorstore.similarity_search(query, k=self.top_k)
            
            if not result:
                if self.verbose:
                    print("No relevant knowledge retrieved")
                return prompt
            
            # 组合检索结果，按来源分组
            source_groups = {}
            for doc in result:
                source = doc.metadata.get("source_file", "Unknown")
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc.page_content)
            
            # 构建增强的提示
            knowledge_sections = []
            for source, contents in source_groups.items():
                section = f"[{source}]:\n" + "\n".join(contents)
                knowledge_sections.append(section)
            
            source_knowledge = "\n\n".join(knowledge_sections)
            
            augmented_prompt = f"""
            使用下面的专业知识来帮助回答查询:
            
            专业知识库:
            {source_knowledge}
            
            查询: 
            {query}
            
            请基于上述专业知识，结合Web测试的最佳实践来回答。
            """
            
            if self.verbose:
                print(f"RAG retrieved {len(result)} chunks from {len(source_groups)} sources")
            
            return augmented_prompt
            
        except Exception as e:
            if self.verbose:
                print(f"RAG retrieval failed: {e}")
            return prompt


class rag_llm_agent(Agent):
    def __init__(self, params):
        self.params = params
        self.verbose = params.get("verbose", False)  # 添加verbose控制参数
        
        # 初始化组件时传递verbose参数
        self.llm = LLMInterface(params, verbose=self.verbose)
        self.retriever = RetrieverInterface(params, verbose=self.verbose)
        self.thinking_kb = ThinkingKnowledgeBase(params, verbose=self.verbose)  # thinking知识库
        self.state_kb = StateKnowledgeBase(params, verbose=self.verbose)  # 新增状态知识库
        self.app_name = params.get("app_name", "Web Testing")
        self.history = []
        self.max_history_length = params.get("max_history_length", 5)

        # 检查是否需要在初始化时重置LLM会话
        if params.get("reset_llm_on_init", True):
            if self.verbose:
                print("Resetting LLM session on init...")
            self.reset_llm_session()

        # 检查是否需要在初始化时清空RAG数据库
        if params.get("clear_rag_on_init", True):
            if self.verbose:
                print("Clearing RAG database on init...")
            self.clear_rag_database()

        if self.verbose:
            print(f"RAG LLM Agent initialized for {self.app_name}")
            print(f"Session ID: {self.llm.session_id[:8]}")

    def clear_rag_database(self):
        """
        清空RAG数据库 - 提供给外部调用的便捷方法
        """
        try:
            if self.verbose:
                print("Clearing all RAG databases...")
            self.retriever.clear_vectorstore()
            self.thinking_kb.clear_thinking_vectorstore()
            self.state_kb.clear_state_vectorstore()
            
            if self.verbose:
                print("All RAG databases cleared successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to clear RAG databases: {e}")

    def reset_llm_session(self):
        """
        重置LLM会话状态 - 确保测试独立性
        """
        try:
            self.llm.reset_session()
            if self.verbose:
                print("LLM session reset successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to reset LLM session: {e}")

    def reset_for_new_test(self):
        """
        为新测试重置Agent状态 - 确保完全独立的测试环境
        """
        if self.verbose:
            print("Resetting agent for new test...")
        
        # 1. 清空历史记录
        self.history = []
        
        # 2. 重置LLM会话状态
        self.reset_llm_session()
        
        # 3. 清空所有RAG数据库（包括状态知识库）
        self.clear_rag_database()
        
        if self.verbose:
            print(f"Agent reset complete - New session: {self.llm.session_id[:8]}")

    def format_action_info(self, action):
        """格式化动作信息"""
        if isinstance(action, ClickAction):
            operation_name = "Click"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'unknown')
            addition_info = getattr(action, 'addition_info', '')
            details = f"Widget text: '{action.text}', type: {action_type}, info: {addition_info}"
        elif isinstance(action, RandomInputAction):
            operation_name = "Input"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'input')
            details = f"Input field: '{action.text}', type: {action_type}"
        elif isinstance(action, RandomSelectAction):
            operation_name = "Select"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'select')
            options = getattr(action, 'options', 'N/A')
            details = f"Select field: '{action.text}', type: {action_type}, options: {options}"
        else:
            operation_name = "UnknownOperation"
            # 安全地获取text属性
            text = getattr(action, 'text', 'Unknown')
            details = f"Widget: '{text}'"

        return f"'{operation_name}' on {details}"

    def detect_login_page(self, action_list, html: str = "", page_title: str = "") -> bool:
        """
        检测是否为登录页面
        """
        # 检查页面标题
        login_title_keywords = [
            "sign in", "login", "log in", "登录", "登入"
        ]
        
        if page_title:
            title_lower = page_title.lower()
            for keyword in login_title_keywords:
                if keyword in title_lower:
                    return True
        
        # 检查HTML内容
        if html:
            html_lower = html.lower()
            if "sign in to github" in html_lower or "github login" in html_lower:
                return True
        
        # 检查动作列表中是否有登录相关的输入字段
        login_field_keywords = [
            "username", "email", "password", "login", "sign in"
        ]
        
        input_fields = [action for action in action_list if isinstance(action, RandomInputAction)]
        login_field_count = 0
        
        for action in input_fields:
            field_text = action.text.lower()
            for keyword in login_field_keywords:
                if keyword in field_text:
                    login_field_count += 1
                    break
        
        # 如果有2个或以上的登录相关字段，认为是登录页面
        return login_field_count >= 2

    def generate_login_focused_prompt(self, action_list, page_context: str, history_str: str) -> str:
        """
        生成专注于登录的提示词
        """
        descriptions = [f"{i}. " + self.format_action_info(a) for i, a in enumerate(action_list)]
        action_descriptions_str = "\n".join(descriptions)
        
        # 识别登录相关的动作
        username_actions = []
        password_actions = []
        login_button_actions = []
        
        for i, action in enumerate(action_list):
            if isinstance(action, RandomInputAction):
                field_text = action.text.lower()
                if any(keyword in field_text for keyword in ["username", "email", "user"]):
                    username_actions.append(i)
                elif any(keyword in field_text for keyword in ["password", "pwd"]):
                    password_actions.append(i)
            elif isinstance(action, ClickAction):
                click_text = action.text.lower()
                if any(keyword in click_text for keyword in ["sign in", "login", "log in", "登录"]):
                    login_button_actions.append(i)
        
        login_suggestions = ""
        if username_actions:
            login_suggestions += f"\n用户名输入字段索引: {username_actions} (输入: Nefelibata-Zhu)"
        if password_actions:
            login_suggestions += f"\n密码输入字段索引: {password_actions} (输入: han19780518)"
        if login_button_actions:
            login_suggestions += f"\n登录按钮索引: {login_button_actions}"
        
        return f"""
        检测到GitHub登录页面！请立即完成登录流程。
        
        {page_context}
        {history_str}
        
        可以操作的界面元素有:
        {action_descriptions_str}
        
        登录操作建议：{login_suggestions}
        
        登录步骤（请严格按顺序执行）：
        1. 找到用户名/邮箱输入字段 → 输入"Nefelibata-Zhu"
        2. 找到密码输入字段 → 输入"han19780518"
        3. 点击"Sign in"登录按钮（避免注册按钮）
        
        重要提醒：
        - 优先完成登录，不要进行其他操作
        - 确保使用正确的凭据格式
        - 避免点击注册相关按钮
        
        请返回一个数字，对应上面列表中动作的索引。
        如果选择输入动作，格式为"索引:文本"。
        例如：用户名字段输入，返回"{username_actions[0] if username_actions else 'X'}:Nefelibata-Zhu"
        
        只返回索引数字或"索引:文本"格式，不要返回任何其他解释。
        """.strip()

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        """
        🚀 增强版决策方法 - 利用多层RAG信息进行智能决策
        """
        action_list = web_state.get_action_list()
        if self.verbose:
            print(f"Available actions: {len(action_list)}")
        if not action_list:
            if self.verbose:
                print("Warning: No available actions")
            return None

        descriptions = [f"{i}. " + self.format_action_info(a) for i, a in enumerate(action_list)]
        action_descriptions_str = "\n".join(descriptions)

        # 获取页面上下文信息
        page_context = ""
        page_title = ""
        if html:
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
            if title_match:
                page_title = title_match.group(1)
                page_context = f"当前页面标题: {page_title}\n"

        # 构建历史记录字符串
        history_str = ""
        if self.history:
            history_str = "最近的操作历史:\n" + "\n".join([f"- {h}" for h in self.history[-self.max_history_length:]])

        # 检测是否为登录页面
        is_login_page = self.detect_login_page(action_list, html, page_title)
        if self.verbose and is_login_page:
            print("Detected login page")

        # 根据是否为登录页面生成不同的提示词
        if is_login_page:
            print("🔐 检测到登录页面，使用专用登录提示词")
            base_prompt = self.generate_login_focused_prompt(action_list, page_context, history_str)
        else:
            # 构建增强的基础提示，包含页面状态分析
            action_count = len(action_list)
            click_count = sum(1 for a in action_list if isinstance(a, ClickAction))
            input_count = sum(1 for a in action_list if isinstance(a, RandomInputAction))
            select_count = sum(1 for a in action_list if isinstance(a, RandomSelectAction))
            
            page_analysis = f"""
当前页面状态分析:
- 总可用动作: {action_count} 个
- 点击元素: {click_count} 个
- 输入字段: {input_count} 个
- 选择框: {select_count} 个
"""
            
            base_prompt = f"""
我们正在测试"{self.app_name}"应用。

{page_context}
{page_analysis}
{history_str}

可以操作的界面元素有:
{action_descriptions_str}

作为Web测试专家，请选择最合适的操作来继续探索和测试应用。考虑以下因素:
1. 当前页面的特点和主要功能
2. 探索新功能和页面路径
3. 测试关键功能流程
4. 发现潜在的bug和边界情况
5. 利用历史测试经验指导决策

请返回一个数字，对应上面列表中动作的索引。
如果选择的动作需要文本输入，请返回索引后跟冒号和输入文本。
例如：选择用户名输入框并输入账号，返回"2:Nefelibata-Zhu"

只返回索引数字或"索引:文本"格式，不要返回任何其他解释。
""".strip()

        try:
            # 🔍 1. 使用传统RAG增强提示（静态知识）
            augmented_prompt = self.retriever.retrieve(base_prompt)

            # 🧠 2. 从thinking知识库检索相关推理经验
            thinking_context = self.thinking_kb.retrieve_relevant_thinking(base_prompt, k=3)

            # 📊 3. 从状态知识库检索相似页面状态（重要改进！）
            state_context = self.state_kb.retrieve_similar_states(page_title, len(action_list), k=2)

            # 🚀 4. 组合所有上下文信息
            context_sections = [augmented_prompt]
            
            if thinking_context:
                context_sections.append(thinking_context)
            
            if state_context:
                context_sections.append(state_context)
            
            final_prompt = "\n\n".join(context_sections)

            if self.verbose:
                rag_enhanced = len(augmented_prompt) > len(base_prompt)
                print(f"Enhancement - RAG: {'✓' if rag_enhanced else '✗'} | Thinking: {'✓' if thinking_context else '✗'} | State: {'✓' if state_context else '✗'}")

            # 5. 调用LLM获取决策和thinking过程
            llm_response = self.llm.chat_with_thinking(final_prompt)
            llm_output = llm_response["content"]
            reasoning = llm_response["reasoning"]

            if self.verbose:
                print(f"LLM output: {llm_output}")

            # 6. 解析LLM输出
            action_index, input_text = self.parse_output(llm_output, len(action_list))

            if action_index is not None and 0 <= action_index < len(action_list):
                selected_action = action_list[action_index]

                # 如果是输入类动作并且有输入文本
                if isinstance(selected_action, RandomInputAction) and input_text:
                    if hasattr(selected_action, 'set_input_text'):
                        selected_action.set_input_text(input_text)

                # 记录操作到历史
                action_description = self.format_action_info(selected_action)
                if input_text:
                    action_description += f" with input: '{input_text}'"
                self.history.append(action_description)

                # 📊 7. 保存完整的页面状态信息到状态知识库（关键改进！）
                self.state_kb.add_page_state(
                    page_title=page_title or "Unknown Page",
                    action_list=action_list,
                    selected_action_index=action_index,
                    reasoning=reasoning
                )

                # 🧠 8. 保存thinking过程到thinking知识库
                self.thinking_kb.add_thinking(
                    prompt=base_prompt,
                    reasoning=reasoning,
                    action_taken=action_description
                )

                # 保持历史记录在限定长度内
                if len(self.history) > self.max_history_length:
                    self.history = self.history[-self.max_history_length:]

                if self.verbose:
                    print(f"Selected action [{action_index}]: {action_description}")
                else:
                    # 在非verbose模式下，只显示最基本的选择信息
                    print(f"Action [{action_index}]: {self.format_action_info(selected_action)}")
                
                return selected_action
            else:
                if self.verbose:
                    print(f"Invalid index {action_index}, using fallback strategy")
                fallback_action = random.choice(action_list)

                # 即使是回退策略，也保存状态和thinking信息
                fallback_description = f"Fallback: {self.format_action_info(fallback_action)}"
                
                self.state_kb.add_page_state(
                    page_title=page_title or "Unknown Page",
                    action_list=action_list,
                    selected_action_index=action_list.index(fallback_action),
                    reasoning=f"解析失败，使用随机回退策略: {reasoning if reasoning else '无推理过程'}"
                )

                if reasoning:
                    self.thinking_kb.add_thinking(
                        prompt=base_prompt,
                        reasoning=reasoning,
                        action_taken=fallback_description
                    )

                return fallback_action

        except Exception as e:
            if self.verbose:
                print(f"Error in decision making: {e}")
            fallback_action = random.choice(action_list)
            fallback_index = action_list.index(fallback_action)

            # 记录错误情况
            error_reasoning = f"执行过程中发生错误: {str(e)}，使用随机策略作为备选"
            error_description = f"Error fallback: {self.format_action_info(fallback_action)}"
            
            # 即使出错也保存状态信息
            self.state_kb.add_page_state(
                page_title=page_title or "Unknown Page",
                action_list=action_list,
                selected_action_index=fallback_index,
                reasoning=error_reasoning
            )
            
            self.thinking_kb.add_thinking(
                prompt=base_prompt,
                reasoning=error_reasoning,
                action_taken=error_description
            )

            return fallback_action

    def parse_output(self, output: str, num_actions: int) -> tuple:
        """
        解析LLM输出，提取动作索引和可能的输入文本
        返回一个元组 (动作索引, 输入文本)，如果没有输入文本则为None
        """
        # 内部定义一个函数，用于去除 ANSI 转义序列
        def remove_ansi(text: str) -> str:
            ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_pattern.sub('', text)

        cleaned_output = remove_ansi(output).strip()

        # 尝试匹配"索引:文本"格式
        input_pattern = re.compile(r'^(\d+):(.+)$', re.DOTALL)
        match = input_pattern.match(cleaned_output)

        if match:
            index_str, input_text = match.groups()
            try:
                index = int(index_str)
                if 0 <= index < num_actions:
                    return index, input_text.strip()
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {index_str}")
                return None, None

        # 尝试直接找出数字
        number_match = re.search(r'^\d+', cleaned_output)
        if number_match:
            try:
                index = int(number_match.group())
                if 0 <= index < num_actions:
                    return index, None
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {number_match.group()}")
                return None, None

        # 尝试从文本中提取最后一个数字作为索引
        numbers = re.findall(r'\d+', cleaned_output)
        if numbers:
            try:
                index = int(numbers[-1])
                if 0 <= index < num_actions:
                    return index, None
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {numbers[-1]}")
                return None, None

        if self.verbose:
            print(f"Failed to parse output: {cleaned_output[:50]}...")
        return None, None


def main():
    """简化的RAG LLM Agent测试"""
    # 测试参数
    test_params = {
        "api_key": "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
        "embedding_token": "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
        "app_name": "Test Application",
        "verbose": True,  # 测试时启用详细输出
        "max_tokens": 512,
        "temperature": 0.7,
        "clear_rag_on_init": True,
        "clear_thinking_on_init": True,
        "clear_state_on_init": True,
        "reset_llm_on_init": True,
    }

    print("RAG LLM Agent Test")
    print("=" * 40)
    
    try:
        # 测试Agent初始化
        agent = rag_llm_agent(test_params)
        print("✓ Agent initialized successfully")
        
        # 测试重置功能
        original_session = agent.llm.session_id
        agent.reset_for_new_test()
        new_session = agent.llm.session_id
        
        print(f"✓ Reset test: {'Success' if original_session != new_session else 'Failed'}")
        
        print("=" * 40)
        print("Test completed successfully!")
        print("\nTo disable verbose output, set params['verbose'] = False")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


if __name__ == "__main__":
    # 只有在直接运行此文件时才执行测试
    main()
